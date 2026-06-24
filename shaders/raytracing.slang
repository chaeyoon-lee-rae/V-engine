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

    image[launchID] = float4(hitValue, 1.0);
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

