#include "external/eva/src/eva-runtime.h"
#include <vector>
#include <iostream>
#include <cstring>

using namespace eva;

static Device device = Runtime::get().device({
    .enableGraphicsQueues = true,
    .enableWindow = true,
    .enableRaytracing = true
    });

static DescriptorPool pool = device.createDescriptorPool({
    .maxTypes = {
        DESCRIPTOR_TYPE::ACCELERATION_STRUCTURE <= 32,
        DESCRIPTOR_TYPE::STORAGE_IMAGE <= 32,
        DESCRIPTOR_TYPE::UNIFORM_BUFFER <= 32,
    },
    .maxSets = 32
    });

static const uint32_t asAlign = device.asBufferOffsetAlignment();
static const uint32_t scratchAlign = device.minAccelerationStructureScratchOffsetAlignment();
static const uint32_t sbtAlign = portable::shaderGroupBaseAlignment;
static const uint32_t sbtRecordAlign = portable::shaderGroupHandleAlignment;
static const uint32_t handleSize = portable::shaderGroupHandleSize;



struct HitgCustomData {
    float color[3];
};

int main()
{
    // =========================================================================
    // [Section 1] Window Creation & Basic Data Definition
    // 렌더링할 윈도우를 생성하고, GPU로 넘길 정점(Vertex), 인덱스(Index), 변환(Transform)
    // 데이터들을 CPU 쪽에 정의합니다.
    // =========================================================================
    Window window = Runtime::get().createWindow({
        .title = "Vulkan Window",
        .width = 800,
        .height = 600,
        .device = device,
        .swapChainImageUsage = IMAGE_USAGE::TRANSFER_DST,
        .swapChainImageFormat = FORMAT::R8G8B8A8_UNORM
        });

    float vertices[][3] = { // 사각형
        { -1.0, -1.0f, 0.0f },
        {  1.0, -1.0f, 0.0f },
        {  1.0,  1.0f, 0.0f },
        { -1.0,  1.0f, 0.0f },
    };
    uint32_t indices[] = { 0, 1, 3, 1, 2, 3 };

    TransformMatrix geoTransforms[] = {
        { { { 1.0f, 0.0f, 0.0f, -2.0f },
            { 0.0f, 1.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f, 0.0f } } },
        { { { 1.0f, 0.0f, 0.0f, 2.0f },
            { 0.0f, 1.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f, 0.0f } } },
    };

    // =========================================================================
    // [Section 2] Geometry Buffer Creation (Vertex, Index, Transform)
    // CPU에 있는 기하학 데이터(정점, 인덱스, 변환 행렬)를 GPU 메모리(Buffer)로 복사합니다.
    // 레이트레이싱 가속 구조체(AS) 빌드 시 읽기 전용으로 사용됩니다.
    // =========================================================================
    Buffer vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        });
    std::memcpy(vertexBuffer.map(), vertices, sizeof(vertices));
    vertexBuffer.unmap();

    Buffer indexBuffer = device.createBuffer({
        .size = sizeof(indices),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        });
    std::memcpy(indexBuffer.map(), indices, sizeof(indices));
    indexBuffer.unmap();

    Buffer geoTransformBuffer = device.createBuffer({
        .size = sizeof(geoTransforms),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        });
    std::memcpy(geoTransformBuffer.map(), geoTransforms, sizeof(geoTransforms));
    geoTransformBuffer.unmap();

    // =========================================================================
    // [Section 3] AS Size Query & Memory Allocation (가속 구조체 크기 구하기 및 버퍼 할당)
    // BLAS(삼각형 데이터)와 TLAS(인스턴스 데이터)를 담을 메모리 크기를 먼저 계산합니다.
    // 크기를 알기 위해 'primitive 개수' 등 메타데이터만 껍데기 Info에 담아 질의(Query)합니다.
    // =========================================================================
    std::vector<AsBuildInfo> blasInfos(1);
    blasInfos[0] = {
        .buildFlags = BUILD_ACCELERATION_STRUCTURE::PREFER_FAST_TRACE,
        .geometryType = GEOMETRY_TYPE::TRIANGLES,
        .primitiveCounts = { 2, 2 }, // 2 geometries, each with 2 triangles
    };

    AsBuildInfo tlasInfo = {
        .buildFlags = BUILD_ACCELERATION_STRUCTURE::PREFER_FAST_TRACE,
        .geometryType = GEOMETRY_TYPE::INSTANCES,
        .primitiveCounts = { 2 }, // 2 instances
    };

    auto blasSize = device.getBuildSizesInfo(blasInfos[0]);
    auto tlasSize = device.getBuildSizesInfo(tlasInfo);

    auto tlasOffset = alignTo(blasSize.accelerationStructureSize, asAlign);
    auto totalAsSize = tlasOffset + tlasSize.accelerationStructureSize;
    auto scratchSize = std::max(blasSize.buildScratchSize, tlasSize.buildScratchSize);

    Buffer asBuffer = device.createBuffer({
        .size = (size_t)totalAsSize,
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_STORAGE | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
        });

    Buffer scratchBuffer = device.createBuffer({
        .size = scratchSize,
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
        });

    // =========================================================================
    // [Section 4] Acceleration Structure Objects Creation
    // 계산된 크기를 바탕으로 할당된 버퍼를 연결하여 가속 구조체(AS) 객체를 생성합니다.
    // 아직 내용은 채워지지 않은 빈 그릇 상태입니다.
    // =========================================================================
    AccelerationStructure blas = device.createAccelerationStructure({
        .asType = ACCELERATION_STRUCTURE_TYPE::BOTTOM_LEVEL,
        .internalBuffer = asBuffer(0),
        .size = blasSize.accelerationStructureSize,
        });

    AccelerationStructure tlas = device.createAccelerationStructure({
        .asType = ACCELERATION_STRUCTURE_TYPE::TOP_LEVEL,
        .internalBuffer = asBuffer(tlasOffset),
        .size = tlasSize.accelerationStructureSize,
        });

    // =========================================================================
    // [Section 5] Instance Buffer Setup (TLAS용 인스턴스 설정)
    // TLAS에 들어갈 각 인스턴스(객체)의 정보(어떤 BLAS를 쓸지, 위치는 어디인지, SBT 오프셋은 무엇인지)
    // 를 정의하고 GPU 버퍼(instanceBuffer)에 업로드합니다.
    // =========================================================================
    std::vector<AccelerationStructureInstance> instances(2);
    instances[0] = {
        .transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 2.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        },
        .instanceCustomIndex = 0,
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = GEOMETRY_INSTANCE::TRIANGLE_FACING_CULL_DISABLE,
        .accelerationStructureReference = blas.deviceAddress()
    };
    instances[1] = {
        .transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, -2.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        },
        .instanceCustomIndex = 0,
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = 2,
        .flags = GEOMETRY_INSTANCE::TRIANGLE_FACING_CULL_DISABLE,
        .accelerationStructureReference = blas.deviceAddress()
    };

    Buffer instanceBuffer = device.createBuffer({
        .size = sizeof(AccelerationStructureInstance) * instances.size(),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
               | BUFFER_USAGE::SHADER_DEVICE_ADDRESS
               | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
        });

    Buffer stagingBuffer = device.createBuffer({
        .size = sizeof(AccelerationStructureInstance) * instances.size(),
        .usage = BUFFER_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        });
    std::memcpy(stagingBuffer.map(), instances.data(), sizeof(AccelerationStructureInstance) * instances.size());
    stagingBuffer.unmap();

    // =========================================================================
    // [Section 6] Build Acceleration Structures (실제 AS 빌드 명령 전송)
    // 앞서 만든 '빈 그릇' AS Info에 진짜 버퍼들(Vertex, Index, Instance)을 연결해주고,
    // 커맨드 버퍼를 통해 GPU에게 가속 구조체를 빌드하라고 명령을 내립니다.
    // =========================================================================
    blasInfos[0].dstAs = blas;
    blasInfos[0].scratchBuffer = scratchBuffer;
    blasInfos[0].inputs = AsBuildInfo::Triangles{
        .vertexCounts = std::vector<uint32_t>(2, std::size(vertices)),
        .eachGeometry = {
            {
                .flags = GEOMETRY::OPAQUE,
                .vertexInput = { vertexBuffer, sizeof(float) * 3 },
                .indexInput = { indexBuffer, sizeof(uint32_t) },
                .transformBuffer = geoTransformBuffer(0, sizeof(TransformMatrix)),
            },
            {
                .flags = GEOMETRY::OPAQUE,
                .vertexInput = { vertexBuffer, sizeof(float) * 3 },
                .indexInput = { indexBuffer, sizeof(uint32_t) },
                .transformBuffer = geoTransformBuffer(sizeof(TransformMatrix), sizeof(TransformMatrix)),
            }
        }
    };

    tlasInfo.dstAs = tlas;
    tlasInfo.scratchBuffer = scratchBuffer;
    tlasInfo.inputs = AsBuildInfo::Instances{
        .instanceInput = instanceBuffer,
    };

    device.newCommandBuffer(queue_graphics)
        .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .buildAccelerationStructures(blasInfos)
        .copyBuffer(stagingBuffer, instanceBuffer)
        .barrier({
            SYNC_SCOPE::ASBUILD_WRITE_AS / scratchBuffer / SYNC_SCOPE::ASBUILD_READ_WRITE_AS,
            SYNC_SCOPE::ASBUILD_WRITE_AS / asBuffer(0, tlasOffset) / SYNC_SCOPE::ASBUILD_READ_AS,
            SYNC_SCOPE::TRANSFER_DST / instanceBuffer / SYNC_SCOPE::ASBUILD_READ,
            })
            .buildAccelerationStructures(tlasInfo)
        .barrier(SYNC_SCOPE::ASBUILD_WRITE_AS / asBuffer(tlasOffset) / SYNC_SCOPE::RAYTRACING_READ_AS)
        .end()
        .submit();

    // =========================================================================
    // [Section 7] Ray Tracing Pipeline & Shader Binding Table (SBT)
    // 레이트레이싱에 사용할 셰이더들(RayGen, Miss, ClosestHit 등)을 묶어 파이프라인을 만들고,
    // 셰이더 실행 함수 포인터 역할을 하는 SBT(Shader Binding Table)를 구성합니다.
    // =========================================================================
    auto rtPipeline = device.createRaytracingPipeline({
        .rgenStage = SpvBlob::readFrom("shaders/raytracing.rgen.spv"),
        .missStages = { SpvBlob::readFrom("shaders/raytracing.rmiss.spv") },
        .hitGroups = {
            {
                .chitStage = SpvBlob::readFrom("shaders/raytracing.rchit.spv"),
            }
        },
        .maxRecursionDepth = 1,
        });

    ShaderGroupHandle handle_hit = rtPipeline.getHitGroupHandle(0);
    uint32_t recordSize = alignTo(handleSize + sizeof(HitgCustomData), sbtRecordAlign);
    uint32_t numRecords = 4;

    Buffer hitGpSbtBuffer = device.createBuffer({
        .size = recordSize * numRecords,
        .usage = BUFFER_USAGE::SHADER_BINDING_TABLE | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        });
    {
        uint8_t* pSbtRecord = (uint8_t*)hitGpSbtBuffer.map();

        HitgCustomData colors[4] = {
            { 0.6f, 0.1f, 0.2f },
            { 0.1f, 0.8f, 0.4f },
            { 0.9f, 0.7f, 0.1f },
            { 0.3f, 0.6f, 0.9f },
        };

        for (uint32_t i = 0; i < 4; i++) {
            std::memcpy(pSbtRecord, &handle_hit, handleSize);
            std::memcpy(pSbtRecord + handleSize, &colors[i], sizeof(HitgCustomData));
            pSbtRecord += recordSize;
        }
        hitGpSbtBuffer.unmap();
    }
    rtPipeline.setHitGroupSbt({ hitGpSbtBuffer, recordSize, numRecords });

    auto queue = device.queue(queue_graphics);

    // =========================================================================
    // [Section 8] Output Image & Descriptor / Sync Objects
    // 레이트레이싱 결과를 기록할 이미지(Storage Image)를 만들고, 
    // 매 프레임마다 변하는 카메라 정보(Uniform Buffer)와 동기화 객체(Fence, Semaphore)를 세팅합니다.
    // =========================================================================
    Image renderImage = device.createImage({
        .format = FORMAT::R8G8B8A8_UNORM,
        .extent = {
            .width = 800,
            .height = 600,
        },
        .usage = IMAGE_USAGE::STORAGE | IMAGE_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
        });

    window.recordPrePresentCommands([&](CommandBuffer cmdBuff, Image swapChainImage) {
        cmdBuff.begin()
            .barrier({
                SYNC_SCOPE::RAYTRACING_WRITE / renderImage / SYNC_SCOPE::TRANSFER_SRC,
                SYNC_SCOPE::NONE / swapChainImage / SYNC_SCOPE::TRANSFER_DST
                })
            .copyImage(renderImage, swapChainImage)
            .barrier({
                SYNC_SCOPE::TRANSFER_SRC / renderImage / SYNC_SCOPE::RAYTRACING_WRITE,
                SYNC_SCOPE::TRANSFER_DST / swapChainImage / SYNC_SCOPE::PRESENT_SRC
                })
            .end();
        });

    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    Buffer uniformBuffer[MAX_FRAMES_IN_FLIGHT];
    float* pMap[MAX_FRAMES_IN_FLIGHT];
    DescriptorSet descSet[MAX_FRAMES_IN_FLIGHT];
    CommandBuffer renderCommandBuffers[MAX_FRAMES_IN_FLIGHT];
    Semaphore onScImageWritable[MAX_FRAMES_IN_FLIGHT];
    Fence fence[MAX_FRAMES_IN_FLIGHT];

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        uniformBuffer[i] = device.createBuffer({
            .size = sizeof(float) * 16,  // vec3 pos + float fov + vec3 X + pad + vec3 Y + pad + vec3 Z + pad
            .usage = BUFFER_USAGE::UNIFORM_BUFFER,
            .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
            });
        pMap[i] = (float*)uniformBuffer[i].map();

        descSet[i] = pool(rtPipeline.descSetLayout(0));
        descSet[i].write({ tlas, renderImage, uniformBuffer[i] });

        renderCommandBuffers[i] = device.newCommandBuffer(queue_graphics)
            .begin()
            .bindPipeline(rtPipeline)
            .bindDescSets({ descSet[i] })
            .traceRays(800, 600)
            .end();

        onScImageWritable[i] = device.createSemaphore();
        fence[i] = device.createFence(true);
    }

    device.newCommandBuffer(queue_graphics)
        .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .barrier(SYNC_SCOPE::NONE / renderImage / SYNC_SCOPE::RAYTRACING_WRITE)
        .end()
        .submit();

    uint32_t currentFrame = 0;

    // =========================================================================
    // [Section 9] Main Render Loop
    // 매 프레임마다 카메라 행렬(Uniform)을 업데이트하고, GPU에 렌더링 커맨드를 제출하며,
    // 화면(Swapchain)에 결과를 출력(Present)합니다.
    // =========================================================================
    while (!window.shouldClose())
    {
        window.pollEvents();

        fence[currentFrame].wait(true);

        auto [writeScImage, onScImagePresentable]
            = window.getNextPresentingContext(onScImageWritable[currentFrame]);

        float* uniformData = pMap[currentFrame];
        {
            uniformData[0] = 0.0f;
            uniformData[1] = 0.0f;
            uniformData[2] = 10.0f;
            uniformData[3] = 60.0f;

            uniformData[4] = 1.0f;
            uniformData[5] = 0.0f;
            uniformData[6] = 0.0f;
            uniformData[7] = 0.0f;

            uniformData[8] = 0.0f;
            uniformData[9] = 1.0f;
            uniformData[10] = 0.0f;
            uniformData[11] = 0.0f;

            uniformData[12] = 0.0f;
            uniformData[13] = 0.0f;
            uniformData[14] = -1.0f;
            uniformData[15] = 0.0f;
        }

        queue << onScImageWritable[currentFrame]
            / (renderCommandBuffers[currentFrame], writeScImage)
            / onScImagePresentable
            << fence[currentFrame];

        window.present(queue);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    queue.waitIdle();

    return 0;
}