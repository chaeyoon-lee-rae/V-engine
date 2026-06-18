// struct VSInput {
//     float3 inPos;
//     float3 inColor;
//     float2 inTexCoord;
// }

// struct UniformBuffer {
//     float4x4 model;
//     float4x4 view;
//     float4x4 proj;
// }

// ConstantBuffer<UniformBuffer> ubo;

// struct VSOutput {
//     float4 pos : SV_Position;
//     float3 fragColor;
//     float2 fragTexCoord;
// };

// Sampler2D texture;

// [shader("vertex")]
// VSOutput vertMain(VSInput input) {
//     VSOutput output;
//     output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPos, 1.0))));
//     output.fragColor = input.inColor;
//     output.fragTexCoord = input.inTexCoord;
//     return output;
// }

// [shader("fragment")]
// float4 fragMain(VSOutput inVert) : SV_Target {
//     return texture.Sample(inVert.fragTexCoord);
// }

// struct VSInput {
//     float2 inPosition;
//     float4 inColor;
// };

// struct VSOutput
// {
//     float4 pos : SV_Position;
//     float pointSize : SV_PointSize;
//     float3 fragColor : COLOR0;
// };

// struct PSInput
// {
//     float4 pos : SV_POSITION;
//     float3 fragColor : COLOR0;
//     float2 pointCoord : SV_PointCoord;
// };

// [shader("vertex")]
// VSOutput vertMain(VSInput input) {
//     VSOutput output;
//     output.pointSize = 14.0;
//     output.pos = float4(input.inPosition, 0.0, 1.0);
//     output.fragColor = input.inColor.rgb;
//     return output;
// }

// [shader("fragment")]
// float4 fragMain(PSInput input) : SV_TARGET {
//     float2 coord = input.pointCoord - float2(0.5);
//     return float4(input.fragColor, 0.5 - length(coord));
// }

// struct Particle {
//     float2 position;
//     float2 velocity;
//     float4 color;
// };

// struct UniformBuffer {
//     float deltaTime;
// };
// ConstantBuffer<UniformBuffer> ubo;

// struct ParticleSSBO {
//     Particle particles;
// };
// StructuredBuffer<ParticleSSBO> particlesIn;
// RWStructuredBuffer<ParticleSSBO> particlesOut;

// [shader("compute")]
// [numthreads(256, 1, 1)]
// void compMain(uint3 threadId: SV_DispatchThreadID)
// {
//     uint index = threadId.x;

//     particlesOut[index].particles.position = particlesIn[index].particles.position + particlesIn[index].particles.velocity.xy * ubo.deltaTime;
//     particlesOut[index].particles.velocity = particlesIn[index].particles.velocity;

//     // Flip movement at window border
//     if ((particlesOut[index].particles.position.x <= -1.0) || (particlesOut[index].particles.position.x >= 1.0)) {
//         particlesOut[index].particles.velocity.x = -particlesOut[index].particles.velocity.x;
//     }
//     if ((particlesOut[index].particles.position.y <= -1.0) || (particlesOut[index].particles.position.y >= 1.0)) {
//         particlesOut[index].particles.velocity.y = -particlesOut[index].particles.velocity.y;
//     }
// }


// ────────────────────────────────────────────────────────────
//  Ray tracing : raytracing.rgen / .miss / .chit 를 Slang으로 번역
//  (GLSL 원본 동작을 그대로 옮김)
// ────────────────────────────────────────────────────────────

// binding 0 : TLAS  (GLSL: accelerationStructureEXT)
[[vk::binding(0)]] RaytracingAccelerationStructure topLevelAS;

// binding 1 : 출력 storage image  (GLSL: layout(rgba8) image2D)
[format("rgba8")]
[[vk::binding(1)]] RWTexture2D<float4> image;

// binding 2 : 카메라 UBO  (GLSL: uniform CameraProperties)
struct CameraProperties {
    float3 cameraPos;
    float  yFov_degree;
};
[[vk::binding(2)]] ConstantBuffer<CameraProperties> g;

// ── ray generation ──
[shader("raygeneration")]
void rgenMain()
{
    const float3 cameraX = float3(1, 0, 0);
    const float3 cameraY = float3(0, -1, 0);
    const float3 cameraZ = float3(0, 0, -1);

    const uint2 launchID   = DispatchRaysIndex().xy;       // GLSL: gl_LaunchIDEXT
    const uint2 launchSize = DispatchRaysDimensions().xy;  // GLSL: gl_LaunchSizeEXT

    const float aspect_y = tan(radians(g.yFov_degree) * 0.5);
    const float aspect_x = aspect_y * float(launchSize.x) / float(launchSize.y);

    const float2 screenCoord = float2(launchID) + float2(0.5);     // 픽셀 중심 (gl_LaunchIDEXT)
    const float2 ndc = screenCoord / float2(launchSize) * 2.0 - 1.0;
    float3 rayDir = ndc.x * aspect_x * cameraX + ndc.y * aspect_y * cameraY + cameraZ;

    float3 hitValue = float3(0.0);   // GLSL: rayPayloadEXT (TraceRay 인자로 전달)

    RayDesc ray;
    ray.Origin    = g.cameraPos;
    ray.TMin      = 0.0;
    ray.Direction = rayDir;
    ray.TMax      = 10000.0;

    TraceRay(topLevelAS,
             RAY_FLAG_FORCE_OPAQUE,  // GLSL: gl_RayFlagsOpaqueEXT
             0xff,                   // cull mask
             0, 1, 0,                // sbtRecordOffset, sbtRecordStride, missIndex
             ray,
             hitValue);              // payload

    image[launchID] = float4(hitValue, 0.0);
}

// ── miss ──
[shader("miss")]
void missMain(inout float3 hitValue)   // GLSL: rayPayloadInEXT vec3 hitValue
{
    hitValue = float3(0.0, 0.0, 0.2);
}

// ── closest hit ──
struct CustomData { float3 color; };
[[vk::shader_record]] ConstantBuffer<CustomData> customData;  // GLSL: layout(shaderRecordEXT) buffer

[shader("closesthit")]
void chitMain(inout float3 hitValue, in BuiltInTriangleIntersectionAttributes attribs)
{
    // GLSL → HLSL/Slang 내장값 매핑 주의:
    //   gl_PrimitiveID            -> PrimitiveIndex()
    //   gl_InstanceID             -> InstanceIndex()   (자동 인덱스)
    //   gl_InstanceCustomIndexEXT -> InstanceID()      (커스텀 값)
    //   gl_GeometryIndexEXT       -> GeometryIndex()
    if (PrimitiveIndex() == 1 &&
        InstanceIndex() == 1 &&
        GeometryIndex() == 1) {
        hitValue = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                          attribs.barycentrics.x,
                          attribs.barycentrics.y);
    }
    else {
        hitValue = customData.color;
    }
}

