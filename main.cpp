#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <random>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <memory>
#	include <vulkan/vulkan_raii.hpp>
#	include <iostream>
#	include <fstream>
#	include <stdexcept>
#else
import vulkan;
#endif
 
constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "assets/models/viking_room.obj";
const std::string TEXTURE_PATH = "assets/textures/viking_room.png";

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr int PARTICLE_COUNT = 8192;
constexpr int SHADER_GROUP_HANDLE_SIZE = 32;

const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

using Mat3x4 = std::array<std::array<float, 4>, 3>;

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription() {
		return { .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex };
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
		return { {
			{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
			{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
			{.location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)},
		} };
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

//struct UniformBufferObject {
//	glm::mat4 model;
//	glm::mat4 view;
//	glm::mat4 proj;
//};

//struct UniformBufferObject {
//	float deltaTime = 1.0f;
//};

struct UniformBufferObject {
	float cameraPos[3];
	float yFov_degree;
};

struct Particle {
	glm::vec2 position;
	glm::vec2 velocity;
	glm::vec4 color;

	static vk::VertexInputBindingDescription getBindingDescription() {
		return { .binding = 0, .stride = sizeof(Particle), .inputRate = vk::VertexInputRate::eVertex };
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
		return { {
			{.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Particle, position)},
			{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(Particle, color)},
		} };
	};
};

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window = nullptr;

	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device device = nullptr;
	uint32_t queueIndex = ~0;
	vk::raii::Queue graphicsQueue = nullptr;
	vk::raii::Queue computeQueue = nullptr;
	vk::raii::SurfaceKHR surface = nullptr;

	vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProps;

	uint32_t frameIndex = 0;
	double lastFrameTime = 0.0;
	double lastTime = 0.0;

	vk::raii::SwapchainKHR swapChain = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::SurfaceFormatKHR swapChainSurfaceFormat;
	vk::Extent2D swapchainExtent;

	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout graphicsPipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;

	vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout computePipelineLayout = nullptr;
	vk::raii::Pipeline comptePipeline = nullptr;

	vk::raii::DescriptorSetLayout rtDescriptorSetLayout = nullptr;	// 1. topLevelAS, 2. outImage, 3. Uniform(Camera)
	vk::raii::PipelineLayout rtPipelineLayout = nullptr;
	vk::raii::Pipeline rtPipeline = nullptr;

	// Raytracing에서는 descriptorSetLayout말고 접근할 수 있는 특별한 문법이 있다 -> shader binding table
	vk::raii::Buffer sbtBuffer = nullptr;
	vk::raii::DeviceMemory sbtBufferMemory = nullptr;
	vk::StridedDeviceAddressRegionKHR rgenSbt{};
	vk::StridedDeviceAddressRegionKHR missSbt{};
	vk::StridedDeviceAddressRegionKHR hitgSbt{};

	vk::raii::Image depthImage = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView depthImageView = nullptr;

	uint32_t mipLevels;
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::Sampler textureSampler = nullptr;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::unordered_map<Vertex, uint32_t> uniqueVertices{};
	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;

	vk::raii::Buffer blasBuffer = nullptr;
	vk::raii::DeviceMemory blasBufferMemory = nullptr;
	vk::raii::AccelerationStructureKHR blas = nullptr;
	vk::DeviceAddress blasAddress;

	vk::raii::Buffer tlasBuffer = nullptr;
	vk::raii::DeviceMemory tlasBufferMemory = nullptr;
	vk::raii::AccelerationStructureKHR tlas = nullptr;

	vk::raii::Image outImage = nullptr;	// raytracing에서 render target이 될 이미지
	vk::raii::DeviceMemory outImageMemory = nullptr;
	vk::raii::ImageView outImageView = nullptr;

	std::vector<vk::raii::Buffer> shaderStorageBuffers;
	std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void *> uniformBuffersMapped;

	vk::raii::DescriptorPool descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	vk::raii::CommandPool commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	std::vector<vk::raii::CommandBuffer> computeCommandBuffers;
	std::vector<vk::raii::CommandBuffer> rtCommandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector <vk::raii::Fence> inFlightFences;

	std::vector<vk::raii::Fence> computeInFlightFences;
	std::vector<vk::raii::Semaphore> computeFinishedSemaphores;
	

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

		lastTime = glfwGetTime();
	}

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		//createSwapchainImageViews();
		//createDescriptorSetLayout();
		//createComputeDescriptorSetLayout();
		createRTDescriptorSetLayout();
		//createGraphicsPipeline();
		//createComputePipeline();
		createRTPipeline();
		createCommandPool();
		//createDepthResources();
		//createTextureImage();
		//createTextureImageView();
		//createTextureSampler();
		//loadModel();
		//createVertexBuffer();
		//createIndexBuffer();
		createBLAS();
		createTLAS();
		createOutImage();

		//createShaderStorageBuffers();
		createUniformBuffers();
		
		createDescriptorPool();
		createDescriptorSets();

		createShaderBindingTable();

		createCommandBuffers();
		createSyncObjects();
	}

	// Instance
	std::vector<const char*> getRequiredInstanceExtensions() {
		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		return extensions;
	}

	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{
			.pApplicationName = "Hello Vulkan",
			.applicationVersion = vk::makeApiVersion(0,1,0,0),
			.pEngineName = "No Engine",
			.engineVersion = vk::makeApiVersion(0,1,0,0),
			.apiVersion = vk::ApiVersion14
		};

		// Layers
		std::vector<char const*> requiredLayers;
		if (enableValidationLayers) {
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto layersProperties = context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layersProperties](auto const& requiredLayer) {
			return std::ranges::none_of(layersProperties,
				[requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
			});
		if (unsupportedLayerIt != requiredLayers.end()) {
			throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
		}

		// Instance Extensions
		auto requiredExtensions = getRequiredInstanceExtensions();

		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto unsupportedExtensionIt = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& requiredExtension) {
			return std::ranges::none_of(extensionProperties,
				[requiredExtension](auto const& extensionProperty) { 
					return strcmp(extensionProperty.extensionName, requiredExtension) == 0; 
				});
			});
		if (unsupportedExtensionIt != requiredExtensions.end())
			throw std::runtime_error("Required GLFW extension not supported: " + std::string(*unsupportedExtensionIt));


		vk::InstanceCreateInfo createInfo{
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
			.ppEnabledLayerNames = requiredLayers.data(),
			.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
			.ppEnabledExtensionNames = requiredExtensions.data(),
		};

		instance = vk::raii::Instance(context, createInfo);
	}

	// Surface
	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	// Physical device
	std::vector<const char*> requiredDeviceExtension = { 
		vk::KHRSwapchainExtensionName,
		//vk::KHRBufferDeviceAddressExtensionName, // 1.2부터 core라 확장 불필요
		vk::KHRDeferredHostOperationsExtensionName,	// not used
		vk::EXTDescriptorIndexingExtensionName,	// not used
		vk::KHRAccelerationStructureExtensionName, // 이걸 쓰려면 위에 3개가 필요함

		vk::KHRSpirv14ExtensionName,
		vk::KHRRayTracingPipelineExtensionName, // 이걸 쓰려면 위에 1개가 필요함
		// extension hierechy가 있음
	};

	bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice)
	{
		// Properties
		bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

		// Queue family
		auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
			return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		bool supportsCompute = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
			return !!(qfp.queueFlags & vk::QueueFlagBits::eCompute);
		});

		// Device extensions
		auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, 
			[&availableDeviceExtensions](auto const& requiredDeviceExtension) {
				return std::ranges::any_of(availableDeviceExtensions, 
				[requiredDeviceExtension](auto const& availableDeviceExtension) {
					return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
				});
			});

		// Device features
		auto features = physicalDevice.template getFeatures2<
			vk::PhysicalDeviceFeatures2, 
			vk::PhysicalDeviceVulkan11Features,
			vk::PhysicalDeviceVulkan12Features,
			vk::PhysicalDeviceVulkan13Features, 
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
			vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
			vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
		>();

		bool supportsRequiredFeatures =
			features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
			features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
			features.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress &&
			features.template get<vk::PhysicalDeviceVulkan12Features>().descriptorIndexing &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
			features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
			features.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure &&
			features.template get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline;

		return supportsVulkan1_3 && 
			   supportsGraphics && 
			   supportsCompute && 
			   supportsAllRequiredExtensions && 
			   supportsRequiredFeatures;
	}

	void pickPhysicalDevice()
	{
		auto physicalDevices = instance.enumeratePhysicalDevices();

		auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const& physicalDevice) {
			return isDeviceSuitable(physicalDevice);
		});

		if (devIter == physicalDevices.end()) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		physicalDevice = *devIter;

		// (RT) shader group handle size
		// Physical device에 쿼리해서 물어봄
		// Vulkan hardware capability viewer를 보고 properties -> extensions -> vk_khr_ray_tracing_pipeline을 보면 됨
		// 실제로 이 값들을 사용해야하기 때문에 쿼리를 해야한다
		auto prop = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
		rtProps = prop.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

		// vulkan spec대로 드라이버가 구현이 되는 건데 스펙을 보면 shader group handle size는 exact하게 32가 되어야한다고 한다?
		// 그래서 32로 강제
		if (rtProps.shaderGroupHandleSize != SHADER_GROUP_HANDLE_SIZE)
			throw std::runtime_error("shaderGroupHandleSize must be 32!");

		std::cout << "Selected GPU : " << static_cast<const char*>(physicalDevice.getProperties().deviceName) << "\n";
	}

	// Logical device
	void createLogicalDevice()
	{
		// Queue
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// get the first index into queueFamilyProperties which supports graphics, compute and present
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); ++qfpIndex) {
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
				(queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
				physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
			throw std::runtime_error("Could not find a queue for graphics, compute and present -> terminating");

		// Feature structures (chained)
		vk::StructureChain<
			vk::PhysicalDeviceFeatures2,
			vk::PhysicalDeviceVulkan11Features,
			vk::PhysicalDeviceVulkan12Features,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
			vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
			vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
		> featureChain = {
			{.features = {.samplerAnisotropy = true}},
			{.shaderDrawParameters = true }, 
			{.descriptorIndexing = true, .bufferDeviceAddress = true},
			{.synchronization2 = true, .dynamicRendering = true},
			{.extendedDynamicState = true },
			{.accelerationStructure = true},
			{.rayTracingPipeline = true},
		};	// now vulkan automatically connects pNext

		// Create a device
		float queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex,
														 .queueCount = 1,
														 .pQueuePriorities = &queuePriority };
		vk::DeviceCreateInfo deviceCreateInfo{
			.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
			.ppEnabledExtensionNames = requiredDeviceExtension.data(),
		};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		graphicsQueue = vk::raii::Queue(device, queueIndex, 0);
		computeQueue = vk::raii::Queue(device, queueIndex, 0);
	}

	// Swap chain
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
	{
		const auto formatIt = std::ranges::find_if(
			availableFormats,
			[](const auto& format) {
				return format.format == vk::Format::eB8G8R8A8Srgb &&
					format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
			});
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) {
			return presentMode == vk::PresentModeKHR::eFifo; }));

		return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) {
			return value == vk::PresentModeKHR::eMailbox; }) ?
			vk::PresentModeKHR::eMailbox :
				vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilites)
	{
		if (capabilites.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilites.currentExtent;
		}

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
			std::clamp<uint32_t>(static_cast<uint32_t>(width), capabilites.minImageExtent.width, capabilites.maxImageExtent.width),
			std::clamp<uint32_t>(static_cast<uint32_t>(height), capabilites.minImageExtent.height, capabilites.maxImageExtent.height),
		};
	}

	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
			minImageCount = surfaceCapabilities.maxImageCount;
		return minImageCount;
	}

	void createSwapChain()
	{
		std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

		std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
		auto availablePresentMode = chooseSwapPresentMode(availablePresentModes);

		vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapchainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
			.surface = *surface,
			.minImageCount = minImageCount,
			.imageFormat = swapChainSurfaceFormat.format,
			.imageColorSpace = swapChainSurfaceFormat.colorSpace,
			.imageExtent = swapchainExtent,
			.imageArrayLayers = 1,
			//.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
			.imageUsage = vk::ImageUsageFlagBits::eTransferDst,
			.imageSharingMode = vk::SharingMode::eExclusive,
			.preTransform = surfaceCapabilities.currentTransform,
			.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = chooseSwapPresentMode(availablePresentModes),
			.clipped = true,
		};

		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	// Image views
	vk::raii::ImageView createImageView(vk::Image const &image, 
										vk::Format format, 
										vk::ComponentMapping component = vk::ComponentMapping(),
										vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor,
										uint32_t mipLevels = 1)
	{
		vk::ImageViewCreateInfo viewInfo{
			.image = image,
			.viewType = vk::ImageViewType::e2D,
			.format = format,
			.components = component,
			.subresourceRange = {
				.aspectMask = aspectFlags,
				.baseMipLevel = 0,
				.levelCount = mipLevels,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		};

		return vk::raii::ImageView(device, viewInfo);
	}

	// 그래픽스 파이프라인에서는 render target, color attachment 그러니까 fragment shader의 output이 되었음
	// 이제 그래픽스 파이프라인 안 쓸거니까 이제 view를 만들 필요가 없음
	// image view가 쓰이는 건 render target으로 쓰거나 texture로 쓸 때 이 2가지 용도로 쓰는데
	// 둘 다 쓰지 않으니 패스하면됨
	//void createSwapchainImageViews()
	//{
	//	assert(swapChainImageViews.empty());

	//	swapChainImageViews.reserve(swapChainImages.size());
	//	for (auto& image : swapChainImages) {
	//		swapChainImageViews.emplace_back(
	//			createImageView(image, 
	//							swapChainSurfaceFormat.format, 
	//							vk::ImageAspectFlagBits::eColor, 1)
	//		);
	//	}
	//}

	// Graphics pipeline
	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		std::vector<char> buffer(file.tellg());

		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

		file.close();
		return buffer;
	}

	[[nodiscard]]
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
		vk::ShaderModuleCreateInfo createInfo{
			.codeSize = code.size() * sizeof(char),
			.pCode = reinterpret_cast<const uint32_t*>(code.data()),
		};

		vk::raii::ShaderModule shaderModule{ device, createInfo };

		return shaderModule;
	}

	void createDescriptorSetLayout()
	{
		std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
			{
				.binding = 0,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eVertex,
			},
			{
				.binding = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eFragment
			}
		}};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data(),
		};

		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createComputeDescriptorSetLayout()
	{
		std::array bindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
			vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data(),
		};

		computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createRTDescriptorSetLayout()
	{
		vk::DescriptorSetLayoutBinding bindings[] = {
			{
				.binding = 0, 
				.descriptorType = vk::DescriptorType::eAccelerationStructureKHR, 
				.descriptorCount = 1, 
				.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
			},
			{
				.binding = 1, 
				.descriptorType = vk::DescriptorType::eStorageImage, 
				.descriptorCount = 1, 
				.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
			},
			{
				.binding = 2,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
			},
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
			.pBindings = bindings
		};

		rtDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createGraphicsPipeline()
	{
		// 1. Shader modules
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = shaderModule,
			.pName = "vertMain"
		};

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = shaderModule,
			.pName = "fragMain"
		};

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };
		
		// Vertex input
		//auto bindingDescription = Vertex::getBindingDescription();
		//auto attributeDescriptions = Vertex::getAttributeDescriptions();
		auto bindingDescription = Particle::getBindingDescription();
		auto attributeDescriptions = Particle::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};

		// Input assembly
		//vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		//	.topology = vk::PrimitiveTopology::eTriangleList,
		//};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
			.topology = vk::PrimitiveTopology::ePointList,
			.primitiveRestartEnable = vk::False,
		};

		// 2. Fixed functions
		// Dynamic States
		std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport,
														vk::DynamicState::eScissor, };
		vk::PipelineDynamicStateCreateInfo dynamicState{
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data(),
		};

		// Viewport & scissor

		//vk::Viewport viewport{
		//	0.0f, 0.0f,
		//	static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height),
		//	0.0f, 1.0f,
		//};
		//vk::Rect2D scissor{ vk::Offset2D{0,0}, swapchainExtent };

		vk::PipelineViewportStateCreateInfo viewportState{
			.viewportCount = 1,
			//.pViewports = &viewport,
			.scissorCount = 1,
			//.pScissors = &scissor,
		};

		// Rasterizer
		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		// Multisampling (disabled for now)
		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable = vk::False,
		};

		// Color blending
		// vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		// 	.blendEnable = vk::False,
		// };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			//.depthTestEnable = vk::True,
			.depthTestEnable = vk::False,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False,
		};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = vk::True,
			.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
			.colorBlendOp = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp = vk::BlendOp::eAdd,
			.colorWriteMask = vk::ColorComponentFlagBits::eR |
							  vk::ColorComponentFlagBits::eG |
							  vk::ColorComponentFlagBits::eB |
							  vk::ColorComponentFlagBits::eA,
		};

		vk::PipelineColorBlendStateCreateInfo colorBlending{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
		};

		// 3. Pipeline layout
		//vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		//	.setLayoutCount = 1,
		//	.pSetLayouts = &*descriptorSetLayout,
		//	.pushConstantRangeCount = 0
		//};

		//graphicsPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
		graphicsPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		// 4. Put all together and create a graphics pipeline
		vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo{
					.stageCount = 2,
					.pStages = shaderStages,
					.pVertexInputState = &vertexInputInfo,
					.pInputAssemblyState = &inputAssembly,
					.pViewportState = &viewportState,
					.pRasterizationState = &rasterizer,
					.pMultisampleState = &multisampling,
					.pDepthStencilState = &depthStencil,
					.pColorBlendState = &colorBlending,
					.pDynamicState = &dynamicState,
					.layout = graphicsPipelineLayout,
					.renderPass = nullptr,
		};

		// +) Dynamic rendering
		vk::Format depthFormat = findDepthFormat();
		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainSurfaceFormat.format,
			.depthAttachmentFormat = depthFormat
		};

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, 
						   vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain{ 
			graphicsPipelineCreateInfo, 
			pipelineRenderingCreateInfo,
		};

		graphicsPipeline = vk::raii::Pipeline(
			device, 
			nullptr, 
			pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>()
		);

	}

	// Compute pipeline
	void createComputePipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eCompute,
			.module = shaderModule,
			.pName = "compMain"
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 1,
			.pSetLayouts = &*computeDescriptorSetLayout,
		};

		computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::ComputePipelineCreateInfo pipelineInfo{
			.stage = computeShaderStageInfo,
			.layout = *computePipelineLayout,
		};

		comptePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
	}

	// RT pipeline
	// 파이프라인이란 쉐이더들의 흐름을 최적화하면서 감싸놓은 것
	void createRTPipeline()
	{
		// pipeline layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 1,
			.pSetLayouts = &*rtDescriptorSetLayout
		};

		rtPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		// 한개의 raygen shader와 여러개의 chit, miss, intersection 쉐이더들이 들어갈 수 있음
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo rgenShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eRaygenKHR,
			.module = shaderModule,
			.pName = "rgenMain"
		};

		vk::PipelineShaderStageCreateInfo chitShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eClosestHitKHR,
			.module = shaderModule,
			.pName = "chitMain"
		};

		vk::PipelineShaderStageCreateInfo missShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eMissKHR,
			.module = shaderModule,
			.pName = "missMain"
		};

		vk::PipelineShaderStageCreateInfo stages[] = { rgenShaderStageInfo, missShaderStageInfo, chitShaderStageInfo };

		// 쉐이더모듈 한 개당 stage하나이다
		// shaderGroups과 stage는 별개이지만 지금은 수를 같게 만들었음
		vk::RayTracingShaderGroupCreateInfoKHR shaderGroups[] = {
			{
				.type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
				.generalShader = 0,	// stages 내 index -> raygen, raygen은 하나 miss shader는 여러개가 있을 수 있는데
				// 각각 하나의 shaderGroup을 가지는 것이다
				.closestHitShader = vk::ShaderUnusedKHR,	// 무조건 unused로 만들어줘야함
				.anyHitShader = vk::ShaderUnusedKHR,
				.intersectionShader = vk::ShaderUnusedKHR,
			},
			{
				.type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
				.generalShader = 1,	// missShader
				.closestHitShader = vk::ShaderUnusedKHR,
				.anyHitShader = vk::ShaderUnusedKHR,
				.intersectionShader = vk::ShaderUnusedKHR,
			},
			{
				.type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
				.generalShader = vk::ShaderUnusedKHR,	// triangleshitgroup 에서는 general shader하고 intersection이 shaderUnused여야한다
				.closestHitShader = 2,
				.anyHitShader = vk::ShaderUnusedKHR,
				.intersectionShader = vk::ShaderUnusedKHR,
			},
		};

		vk::RayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{
			.stageCount = sizeof(stages) / sizeof(stages[0]),
			.pStages = stages,
			.groupCount = sizeof(shaderGroups) / sizeof(shaderGroups[0]),	// (RT)
			.pGroups = shaderGroups,										// (RT)
			.maxPipelineRayRecursionDepth = 1,								// (RT)
			.layout = rtPipelineLayout,
		};
		// RT에만 들어가는 fields; maxPipelineRayRecursionDepth -> 재귀호출을 몇번이나 할 수 있나?
		// GPU 프로그램은 함수 호출 때 CPU와 달리 스택이 없다
		// inline 함수처럼 동작을 한다
		// 스택 이런 것보다 최적화가 잘 되어 있다
		// 매개변수로 넘겨줄 때 레지스터의 그 메모리 위치는 그대로 있고 inline 함수처럼 들어가서 최적화가 잘 되는 것
		// 대신에 재귀호출이 불가능했다... 지금은 재귀호출을 가능하게 만들긴했음

		rtPipeline = device.createRayTracingPipelineKHR(nullptr, nullptr, rtPipelineCreateInfo);
	}

	struct ShaderGroupHandle {
		uint8_t data[SHADER_GROUP_HANDLE_SIZE];		// 1byte * 32 = 32 bytes
	};

	struct HitgCustomData {
		float color[3];
	};

	/*
	In the vulkan spec,
	[VUID-vkCmdTraceRaysKHR-stride-03686] pMissShaderBindingTable->stride must be a multiple of VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleAlignment
	[VUID-vkCmdTraceRaysKHR-stride-03690] pHitShaderBindingTable->stride must be a multiple of VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleAlignment
	[VUID-vkCmdTraceRaysKHR-pRayGenShaderBindingTable-03682] pRayGenShaderBindingTable->deviceAddress must be a multiple of VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupBaseAlignment
	[VUID-vkCmdTraceRaysKHR-pMissShaderBindingTable-03685] pMissShaderBindingTable->deviceAddress must be a multiple of VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupBaseAlignment
	[VUID-vkCmdTraceRaysKHR-pHitShaderBindingTable-03689] pHitShaderBindingTable->deviceAddress must be a multiple of VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupBaseAlignment

	As shown in the vulkan spec 40.3.1. Indexing Rules,
		pHitShaderBindingTable->deviceAddress + pHitShaderBindingTable->stride × (
		instanceShaderBindingTableRecordOffset + geometryIndex × sbtRecordStride + sbtRecordOffset )
	*/

	void createShaderBindingTable()
	{
		auto alignTo = [](auto value, auto alignment) -> decltype(value) {	// alignment는 2의 승수여야함
			return (value + (decltype(value))alignment - 1) & ~((decltype(value))alignment - 1);
		};

		const uint32_t handleSize = SHADER_GROUP_HANDLE_SIZE;
		const uint32_t groupCount = 3;	// 편의상 하드코딩
		
		auto handles = rtPipeline.getRayTracingShaderGroupHandlesKHR<ShaderGroupHandle>(
			0, groupCount, groupCount * handleSize
		);	// record 하나당 하나의 handle이 들어가야함, 총 3개니까 handles vector는 총 32 * 3 = 96 bytes

		ShaderGroupHandle rgenHandle = handles[0];
		ShaderGroupHandle missHandle = handles[1];
		ShaderGroupHandle hitgHandle = handles[2];

		// shaderGroupHandleAlignment와 shaderGroupBaseAlignment 쿼리해서 써야함
		// 실제 stride는 record의 byte 길이
		// shaderGroupHandleAlignment -> 32 max
		// shaderGroupBaseAlignment -> 64 max
		const uint32_t rgenStride = alignTo(handleSize, rtProps.shaderGroupHandleAlignment);
		rgenSbt = { 0, rgenStride, rgenStride };	// device address, stride, size

		const uint64_t missOffset = alignTo(rgenSbt.size, rtProps.shaderGroupBaseAlignment);
		const uint32_t missStride = alignTo(handleSize, rtProps.shaderGroupHandleAlignment);
		missSbt = { 0, missStride, missStride };	// miss shader가 만약 여러개면 size가 늘어날 것

		const uint32_t hitgCustomDataSize = sizeof(HitgCustomData);
		const uint32_t geometryCount = 4;	// 각각의 geometry, geometry마다 record를 하나씩 줄 것임
		const uint64_t hitgOffset = alignTo(missOffset + missSbt.size, rtProps.shaderGroupBaseAlignment);
		const uint32_t hitgStride = alignTo(handleSize + hitgCustomDataSize, rtProps.shaderGroupHandleAlignment);	// 32 + 12 = 44 -> 64로 align됨
		hitgSbt = { 0, hitgStride, hitgStride * geometryCount };	// size는 stride의 배수
		// 보통 material마다 group을 만듦

		const uint64_t sbtSize = hitgOffset + hitgSbt.size;
		std::tie(sbtBuffer, sbtBufferMemory) = createBuffer(
			sbtSize,
			vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent	// 편의상, 실제 implementation 때는 device local로 올릴 것
		);

		auto sbtAddress = device.getBufferAddress({ .buffer = sbtBuffer });
		if (sbtAddress != alignTo(sbtAddress, rtProps.shaderGroupBaseAlignment))	// device address는 shaderGroupBaseAlignment에 align되어야함!
			throw std::runtime_error("It will not be happened!");					// getBufferAddress가 shaderGroupBaseAlignment에 align되어야한다는 것인데, 내가 찾아봤을 땐 그러한 스펙은 찾을 수 없지만 동작은 잘됨
		rgenSbt.deviceAddress = sbtAddress;
		missSbt.deviceAddress = sbtAddress + missOffset;
		hitgSbt.deviceAddress = sbtAddress + hitgOffset;

		uint8_t* dst = static_cast<uint8_t*>(sbtBufferMemory.mapMemory(0, sbtSize));
		{
			*(ShaderGroupHandle*) dst = rgenHandle;		// 이렇게 강제형변환하는 것이 memcpy보다 덜 복잡하기도하고 빠르기도 하다
			*(ShaderGroupHandle*)(dst + missOffset) = missHandle;

			*(ShaderGroupHandle*)(dst + hitgOffset + 0 * hitgStride) = hitgHandle;
			*(HitgCustomData*	)(dst + hitgOffset + 0 * hitgStride + handleSize) = { 0.6f, 0.1f, 0.2f }; // Deep Red Wine
			*(ShaderGroupHandle*)(dst + hitgOffset + 1 * hitgStride) = hitgHandle;
			*(HitgCustomData*	)(dst + hitgOffset + 1 * hitgStride + handleSize) = { 0.1f, 0.8f, 0.4f }; // Emerald Green
			*(ShaderGroupHandle*)(dst + hitgOffset + 2 * hitgStride) = hitgHandle;
			*(HitgCustomData*	)(dst + hitgOffset + 2 * hitgStride + handleSize) = { 0.9f, 0.7f, 0.1f }; // Golden Yellow
			*(ShaderGroupHandle*)(dst + hitgOffset + 3 * hitgStride) = hitgHandle;
			*(HitgCustomData*	)(dst + hitgOffset + 3 * hitgStride + handleSize) = { 0.3f, 0.6f, 0.9f }; // Dawn Sky Blue
		}	
		// 사실 stride의 크기를 똑같이 만들거나 (가장 큰 거에 맞춰서) 아니면 SBT를 3개(그러니까 buffer 3개)를 만드는 더 쉬운 방법도 있다
		sbtBufferMemory.unmapMemory();
	}


	vk::Format findSupportedFormat(
		const std::vector<vk::Format>& candidates, 
		vk::ImageTiling tiling, 
		vk::FormatFeatureFlags features)
	{
		for (const auto format : candidates) {
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);

			if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
				return format;
			else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
				return format;
		}

		throw std::runtime_error("failed to find supported format!");
	}

	vk::Format findDepthFormat() 
	{
		return findSupportedFormat(
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment
		);
	}

	bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
	}

	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();

		std::tie(depthImage, depthImageMemory) = createImage(
			swapchainExtent.width, swapchainExtent.height,
			1,
			depthFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal);
		depthImageView = createImageView(depthImage, depthFormat, {}, vk::ImageAspectFlagBits::eDepth, 1);
	}

	// Images
	vk::raii::CommandBuffer beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool, // TODO: generalize commandPool
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1
		};

		vk::raii::CommandBuffer commandBuffer = std::move(
			vk::raii::CommandBuffers(device, allocInfo).front()
		);

		vk::CommandBufferBeginInfo beginInfo{
			.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
		};
		commandBuffer.begin(beginInfo);

		return std::move(commandBuffer);
	}

	void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo{
			.commandBufferCount = 1,
			.pCommandBuffers = &*commandBuffer
		};
		graphicsQueue.submit(submitInfo, nullptr); // TODO: generalize queue
		graphicsQueue.waitIdle(); // TODO: use fence
	}

	std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(
		uint32_t width, uint32_t height,
		uint32_t mipLevels,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties)
	{
		vk::ImageCreateInfo imageInfo{
			.imageType = vk::ImageType::e2D,
			.format = format,
			.extent = {width, height, 1},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = vk::SampleCountFlagBits::e1,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive
		};

		vk::raii::Image image = vk::raii::Image(device, imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
		};
		vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);

		return { std::move(image), std::move(imageMemory) };
	}

	void copyBufferToImage(vk::raii::CommandBuffer& commandBuffer,
		const vk::raii::Buffer& buffer,
		vk::raii::Image& image,
		uint32_t width,
		uint32_t height)
	{
		vk::BufferImageCopy region{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
			.imageOffset = {0,0,0},
			.imageExtent = {width, height, 1},
		};

		commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
	}

	void generateMipmaps(vk::raii::Image& image,
		vk::Format imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels)
	{
		vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
		if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
			throw std::runtime_error("texture image format does not support linear blitting!");

		vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier{
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image = image };

		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; ++i) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eTransfer,
				{}, {}, {},
				barrier);

			vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
			offsets[0] = vk::Offset3D(0, 0, 0);
			offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);

			dstOffsets[0] = vk::Offset3D(0, 0, 0);
			dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);

			vk::ImageBlit blit = {
				.srcSubresource = {},
				.srcOffsets = offsets,
				.dstSubresource = {},
				.dstOffsets = dstOffsets
			};

			blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
			blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

			commandBuffer.blitImage(image,
				vk::ImageLayout::eTransferSrcOptimal,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				{ blit },
				vk::Filter::eLinear);

			barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eFragmentShader,
				{}, {}, {},
				barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{}, {}, {},
			barrier);

		endSingleTimeCommands(std::move(commandBuffer));
	}

	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels)
			throw std::runtime_error("failed to load texture image!");

		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		auto [stagingBuffer, stagingBufferMemory] = createBuffer(
			imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, pixels, imageSize);
		stagingBufferMemory.unmapMemory();

		stbi_image_free(pixels);

		std::tie(textureImage, textureImageMemory) = createImage(
			texWidth, texHeight,
			mipLevels,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc | 
			vk::ImageUsageFlagBits::eTransferDst | 
			vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
		transitionImageLayout(commandBuffer, 
							  textureImage, 
							  vk::ImageLayout::eUndefined, 
							  vk::ImageLayout::eTransferDstOptimal, 
							  vk::ImageAspectFlagBits::eColor,
							  mipLevels);
		copyBufferToImage(commandBuffer, stagingBuffer, textureImage, texWidth, texHeight);
		endSingleTimeCommands(std::move(commandBuffer));

		generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(*textureImage, 
										   vk::Format::eR8G8B8A8Srgb, 
										   {},
										   vk::ImageAspectFlagBits::eColor, 
										   mipLevels);
	}

	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

		vk::SamplerCreateInfo samplerInfo{
			.magFilter = vk::Filter::eLinear,
			.minFilter = vk::Filter::eLinear,
			.mipmapMode = vk::SamplerMipmapMode::eLinear,
			.addressModeU = vk::SamplerAddressMode::eRepeat,
			.addressModeV = vk::SamplerAddressMode::eRepeat,
			.addressModeW = vk::SamplerAddressMode::eRepeat,
			.mipLodBias = 0.0f,
			.anisotropyEnable = vk::True,
			.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
			.compareEnable = vk::False,
			.compareOp = vk::CompareOp::eAlways,
			.minLod = 0.0f,
			.maxLod = vk::LodClampNone,
			.borderColor = vk::BorderColor::eIntOpaqueBlack,
			.unnormalizedCoordinates = vk::False,
		};

		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	// Loading a model
	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
			throw std::runtime_error(warn + err);

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2],
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				auto [it, inserted] = uniqueVertices.insert({vertex, static_cast<uint32_t>(vertices.size())});
				if (inserted)
					vertices.push_back(vertex);
				indices.push_back(it->second);
			}
		}
	}

	// Vertex buffer
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) 
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties)==properties)
				return i;
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, 
																	 vk::BufferUsageFlags usage, 
																	 vk::MemoryPropertyFlags properties) 
	{
		vk::BufferCreateInfo bufferInfo{
			.size = size,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive,
		};

		vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);

		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

		vk::MemoryAllocateInfo memoryAllocateInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(
				memRequirements.memoryTypeBits,
				properties)
		};

		vk::MemoryAllocateFlagsInfo flagsInfo{};
		if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
		{
			flagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
			memoryAllocateInfo.pNext = &flagsInfo;
		}

		vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);

		buffer.bindMemory(*bufferMemory, 0);

		return { std::move(buffer), std::move(bufferMemory) };
	}

	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
	{
		auto commandCopyBuffer = beginSingleTimeCommands();
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		endSingleTimeCommands(std::move(commandCopyBuffer));
	}

	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		std::tie(vertexBuffer, vertexBufferMemory) = createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	// Index buffer
	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, indices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		std::tie(indexBuffer, indexBufferMemory) = createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	// Storage Buffers
	void createShaderStorageBuffers()
	{
		shaderStorageBuffers.clear();
		shaderStorageBuffersMemory.clear();

		// 파티클 초기화
		std::default_random_engine rndEngine((unsigned)time(nullptr));
		std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

		// 원형 형태의 초기 파티클 위치 설정
		std::vector<Particle> particles(PARTICLE_COUNT);
		for (auto& particle : particles) {
			float r = 0.25f * sqrtf(rndDist(rndEngine));
			float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
			float x = r * cosf(theta) * HEIGHT / WIDTH;
			float y = r * sinf(theta);
			particle.position = glm::vec2(x, y);
			particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
			particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
		}

		vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

		auto [stagingbuffer, stagingbufferMemory] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);

		void* dataStaging = stagingbufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, particles.data(), bufferSize);
		stagingbufferMemory.unmapMemory();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			auto [shaderStorageBufferTemp, shaderStorageBufferTempMemory] = createBuffer(
				bufferSize,
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eDeviceLocal
			);

			copyBuffer(stagingbuffer, shaderStorageBufferTemp, bufferSize);
			shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
			shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferTempMemory));
		}
	}

	// (RT) BLAS
	void createBLAS()	// geometry들로 이루어짐 -> TLAS는 instance들로 이루어짐
	{
		// 일단 하드코딩으로 geometry 만들어줌
		float vertices[][3] = { // 사각형
			{ -1.0, -1.0f, 0.0f },
			{  1.0, -1.0f, 0.0f },
			{  1.0,  1.0f, 0.0f },
			{ -1.0,  1.0f, 0.0f },
		};
		uint32_t indices[] = { 0, 1, 3, 1, 2, 3 };

		vk::TransformMatrixKHR geoTransforms[] = {	// 사각형에 적용할 transform
			{.matrix = Mat3x4{{	// 왼쪽으로 이동
				{{ 1.0f, 0.0f, 0.0f, -2.0f }},
				{{ 0.0f, 1.0f, 0.0f, 0.0f }},
				{{ 0.0f, 0.0f, 1.0f, 0.0f }},
			}}},
			{.matrix = Mat3x4{{	// 오른쪽으로 이동
				{{ 1.0f, 0.0f, 0.0f, 2.0f }},
				{{ 0.0f, 1.0f, 0.0f, 0.0f }},
				{{ 0.0f, 0.0f, 1.0f, 0.0f }},
			}}},
		};

		// 편의성을 위해서 일단 host visible, host coherent로 만듦
		auto [vertexBuffer, vertexBufferMem] = createBuffer(
			sizeof(vertices),
			vk::BufferUsageFlagBits::eShaderDeviceAddress |
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
		);

		auto [indexBuffer, indexBufferMem] = createBuffer(
			sizeof(indices),
			vk::BufferUsageFlagBits::eShaderDeviceAddress |
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
		);

		auto [geoTransformBuffer, geoTransformBufferMem] = createBuffer(
			sizeof(geoTransforms),
			vk::BufferUsageFlagBits::eShaderDeviceAddress |
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
		);

		void* data = vertexBufferMem.mapMemory(0, sizeof(vertices));
		memcpy(data, vertices, sizeof(vertices));
		vertexBufferMem.unmapMemory();

		data = indexBufferMem.mapMemory(0, sizeof(indices));
		memcpy(data, indices, sizeof(indices));
		indexBufferMem.unmapMemory();

		data = geoTransformBufferMem.mapMemory(0, sizeof(geoTransforms));
		memcpy(data, geoTransforms, sizeof(geoTransforms));
		geoTransformBufferMem.unmapMemory();

		vk::AccelerationStructureGeometryKHR geometry0{
			//geometry type: triangles, aabbs -> intersection shader쓸 때, instances
			.geometryType = vk::GeometryTypeKHR::eTriangles,
			.geometry = {
				.triangles = {
					.vertexFormat = vk::Format::eR32G32B32Sfloat,
					.vertexData = {.deviceAddress = device.getBufferAddress({.buffer = vertexBuffer})},	// getBufferAddress?
					.vertexStride = sizeof(vertices[0]),
					.maxVertex = sizeof(vertices) / sizeof(vertices[0]) - 1,
					.indexType = vk::IndexType::eUint32,
					.indexData = {.deviceAddress = device.getBufferAddress({.buffer = indexBuffer})},
					.transformData = {.deviceAddress = device.getBufferAddress({.buffer = geoTransformBuffer})},
				}
			},
			.flags = vk::GeometryFlagBitsKHR::eOpaque,
		};

		vk::AccelerationStructureGeometryKHR geometries[] = { geometry0, geometry0 };

		uint32_t triangleCount0 = sizeof(indices) / (sizeof(indices[0]) * 3);
		uint32_t triangleCounts[] = { triangleCount0, triangleCount0 };

		// pre-build를 해줄 것임
		// 원래 info 안에 많은 정보가 필요한데, pre-build할 때에는 아래 정보만 세팅하면 됨
		vk::AccelerationStructureBuildGeometryInfoKHR buildBlasInfo{
			.type = vk::AccelerationStructureTypeKHR::eBottomLevel,
			.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
			.geometryCount = sizeof(geometries) / sizeof(geometries[0]),
			.pGeometries = geometries,
		};

		auto requiredSize = device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice,	// AS 빌드를 어디서 할 것인가?, DX에서는 무조건 GPU에서만 되는데, vulkan에서는 cpu에서도 가능함
			buildBlasInfo,
			triangleCounts);

		std::tie(blasBuffer, blasBufferMemory) = createBuffer(
			requiredSize.accelerationStructureSize,
			vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal	// GPU가 알아서함, memcpy 이런거 해주지 않음
		);

		auto [scratchBuffer, scratchBufferMem] = createBuffer(	// 추가적인 용량?
			requiredSize.buildScratchSize,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		// BLAS 핸들 만들기
		{
			vk::AccelerationStructureCreateInfoKHR asCreateInfo{
				.buffer = blasBuffer,
				.size = requiredSize.accelerationStructureSize,
				.type = vk::AccelerationStructureTypeKHR::eBottomLevel,	// blas임을 알려줌
			};

			blas = device.createAccelerationStructureKHR(asCreateInfo); // std::move해줘야하나?
			blasAddress = device.getAccelerationStructureAddressKHR({ .accelerationStructure = blas });
		}

		// prebuild에서 사이즈도 얻어오고 필요한 버퍼를 만들어둠
		// 이제 준비과정이 끝났으니 실제로 빌드를 진행
		{
			auto commandBuffer = beginSingleTimeCommands();	// GPU에서 수행되어야하니 commandbuffer 생성
			{
				// 이제 buildBlasInfo에 실제 빌드에 들어가야할 데이터를 넣어준다
				buildBlasInfo.dstAccelerationStructure = blas;
				buildBlasInfo.scratchData.deviceAddress = device.getBufferAddress({ .buffer = scratchBuffer });
				// buildinfo에 들어간 geometry count가 2개이다
				// 각 geometry에 해당하는 rangeinfo를 넣어주어야함
				vk::AccelerationStructureBuildRangeInfoKHR buildBlasRangeInfo[] = {
					{
						.primitiveCount = triangleCounts[0],
						.transformOffset = 0,
					},
					{
						.primitiveCount = triangleCounts[1],
						.transformOffset = sizeof(geoTransforms[0]),
					},
				};

				vk::AccelerationStructureBuildGeometryInfoKHR buildBlasInfos[] = { buildBlasInfo };
				const vk::AccelerationStructureBuildRangeInfoKHR* buildBlasRangeInfos[] = { buildBlasRangeInfo };
				// 여러개의 build acceleration Structures를 만들 수 있도록 해줌
				commandBuffer.buildAccelerationStructuresKHR(buildBlasInfos, buildBlasRangeInfos);
			}
			endSingleTimeCommands(std::move(commandBuffer));
		}
	}

	// (RT) TLAS
	void createTLAS()
	{
		// 이거는 instance마다 가지고 있는 transforms
		vk::TransformMatrixKHR insTransforms[] = {
			{.matrix = Mat3x4{{
				{{1.0f, 0.0f, 0.0f, 0.0f}},
				{{0.0f, 1.0f, 0.0f, 2.0f}},
				{{0.0f, 0.0f, 1.0f, 0.0f}},
			}}},
			{.matrix = Mat3x4{{
				{{1.0f, 0.0f, 0.0f,  0.0f}},
				{{0.0f, 1.0f, 0.0f, -2.0f}},
				{{0.0f, 0.0f, 1.0f,  0.0f}},
			}}},
		};

		// Blas를 내포함, 참조하고 있음
		vk::AccelerationStructureInstanceKHR instance0{
			.mask = 0xFF,
			.instanceShaderBindingTableRecordOffset = 0,
			.flags = static_cast<VkGeometryInstanceFlagsKHR>(vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable),
			.accelerationStructureReference = blasAddress,
		};

		// 첫번째 사각형 2개를 참조하는 instance는 위로 2칸 올리고 다른 isntance는 밑으로 2칸 내리는 것
		vk::AccelerationStructureInstanceKHR instanceData[] = { instance0, instance0 };
		instanceData[0].transform = insTransforms[0];
		instanceData[1].transform = insTransforms[1];
		instanceData[1].instanceShaderBindingTableRecordOffset = 2;

		// 마찬가지로 버퍼를 만들어준다 
		auto [instanceBuffer, instanceBufferMem] = createBuffer(
			sizeof(instanceData),
			vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,// 위 BLAS에서는 vertex, index, geoTransform이 input이면 TLAS에서는 instance가 input buffer가 된다
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void* data = instanceBufferMem.mapMemory(0, sizeof(instanceData));
		memcpy(data, instanceData, sizeof(instanceData));
		instanceBufferMem.unmapMemory();

		// 이제 TLAS 과정 시작
		// BLAS랑 거의 비슷한 구조
		vk::AccelerationStructureGeometryKHR instances{
			.geometryType = vk::GeometryTypeKHR::eInstances,
			.geometry = {	// geometry union이다, 
				.instances = {	// 아까는 triangles에 값을 넣어줬는데 지금은 instances이다
					.data = {.deviceAddress = device.getBufferAddress({.buffer = instanceBuffer})}
				}
			},
			.flags = vk::GeometryFlagBitsKHR::eOpaque
		};

		const uint32_t instanceCount = sizeof(instanceData) / sizeof(instanceData[0]);	// 2

		vk::AccelerationStructureBuildGeometryInfoKHR buildTlasInfo{
			.type = vk::AccelerationStructureTypeKHR::eTopLevel,
			.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
			.geometryCount = 1,	// 여기서 BLAS geometry랑은 다르다, 헷갈리면 안됨, 여기서는 instance가 한개의 geometry이다. 그리고 vulkan 스펙에서 tlas는 geometry count는 1이어야한다
			.pGeometries = &instances
		};

		// 마찬가지로 prebuild를 해서 필요한 공간을 계산
		auto requiredSize = device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice, 
			buildTlasInfo, instanceCount);

		std::tie(tlasBuffer, tlasBufferMemory) = createBuffer(
			requiredSize.accelerationStructureSize,
			vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		auto [scratchBuffer, scratchBufferMem] = createBuffer(
			requiredSize.buildScratchSize,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		// TLAS 핸들 만들기 
		{
			vk::AccelerationStructureCreateInfoKHR asCreateInfo{
				.buffer = tlasBuffer,
				.size = requiredSize.accelerationStructureSize,
				.type = vk::AccelerationStructureTypeKHR::eTopLevel
			};

			tlas = device.createAccelerationStructureKHR(asCreateInfo);
		}

		{
			auto commandBuffer = beginSingleTimeCommands();
			{
				buildTlasInfo.dstAccelerationStructure = tlas;
				buildTlasInfo.scratchData.deviceAddress = device.getBufferAddress({ .buffer = scratchBuffer });

				vk::AccelerationStructureBuildRangeInfoKHR buildTlasRangeInfo = { .primitiveCount = instanceCount };	// BLAS에서는 triangle의 개수였음
				commandBuffer.buildAccelerationStructuresKHR(buildTlasInfo, &buildTlasRangeInfo);
			}
			endSingleTimeCommands(std::move(commandBuffer));
		}

		// 지금까지 raytracing이 동작하는 scene을 만들어줌 -> BVH 구조를 만들어준것
	}

	// (RT) raytracing에서 출력 대상이 될 Image
	void createOutImage()
	{
		vk::Format format = vk::Format::eR8G8B8A8Unorm;	// issue: swapchain format이랑 같아야하는데 swapchain format인 BGRA_SRGB는 storage bit를 지원하지 않음(optimal tiling)

		std::tie(outImage, outImageMemory) = createImage(
			WIDTH, HEIGHT,
			1,
			format,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,	// 그래픽스에서는 color여야지 쓸 수 있고, comput/rt에서는 storage image에 쓸 수 있다
			vk::MemoryPropertyFlagBits::eDeviceLocal);

		outImageView = createImageView(outImage, 
			format, 
			{ .r = vk::ComponentSwizzle::eB, .b = vk::ComponentSwizzle::eR }, 
			vk::ImageAspectFlagBits::eColor, 1);

		auto commandBuffer = beginSingleTimeCommands();
		{
			// compute/rt 에서는 쓰기 전용 layout이 따로 없어서 general로 써야한다
			transitionImageLayout(commandBuffer,
				outImage,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eGeneral,
				{}, {},
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::ImageAspectFlagBits::eColor);
		}
		endSingleTimeCommands(std::move(commandBuffer));
	}

	// Descriptor sets
	void createUniformBuffers()
	{
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
			auto [buffer, bufferMem] = createBuffer(bufferSize, 
													vk::BufferUsageFlagBits::eUniformBuffer, 
													vk::MemoryPropertyFlagBits::eHostVisible | 
													vk::MemoryPropertyFlagBits::eHostCoherent);

			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
		}
	}

	void createDescriptorPool()
	{
		//std::array< vk::DescriptorPoolSize, 2> poolSize{{
		//	{
		//		.type = vk::DescriptorType::eUniformBuffer,
		//		.descriptorCount = MAX_FRAMES_IN_FLIGHT,
		//	},
		//	{
		//		.type = vk::DescriptorType::eCombinedImageSampler,
		//		.descriptorCount = MAX_FRAMES_IN_FLIGHT,
		//	},
		//}};

		//std::array poolSize{
		//	vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
		//	vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)
		//};

		vk::DescriptorPoolSize poolSizes[] = {
			{
				.type = vk::DescriptorType::eAccelerationStructureKHR,
				.descriptorCount = MAX_FRAMES_IN_FLIGHT,
			},
			{
				.type = vk::DescriptorType::eStorageImage,
				.descriptorCount = MAX_FRAMES_IN_FLIGHT,
			},
			{
				.type = vk::DescriptorType::eUniformBuffer,
				.descriptorCount = MAX_FRAMES_IN_FLIGHT,
			},
		};

		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]),
			.pPoolSizes = poolSizes,
		};

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	void createDescriptorSets()
	{
		//std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
		//std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *rtDescriptorSetLayout);

		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data(),
		};

		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			//vk::DescriptorImageInfo imageInfo{
			//	.sampler = textureSampler,
			//	.imageView = textureImageView,
			//	.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			//};

			//vk::DescriptorBufferInfo storageBufferInfoLastFrame(
			//	shaderStorageBuffers[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
			//vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
			//	shaderStorageBuffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);

			//std::array<vk::WriteDescriptorSet, 2> descriptorWrite{{
			//	{
			//		.dstSet = descriptorSets[i],
			//		.dstBinding = 0,
			//		.dstArrayElement = 0,
			//		.descriptorCount = 1,
			//		.descriptorType = vk::DescriptorType::eUniformBuffer,
			//		.pBufferInfo = &bufferInfo,
			//	},
			//	{
			//		.dstSet = descriptorSets[i],
			//		.dstBinding = 1,
			//		.dstArrayElement = 0,
			//		.descriptorCount = 1,
			//		.descriptorType = vk::DescriptorType::eCombinedImageSampler,
			//		.pImageInfo = &imageInfo,
			//	},
			//}};


			// AS structure
			vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{
				.accelerationStructureCount = 1,
				.pAccelerationStructures = &*tlas,
			};

			// Out image
			vk::DescriptorImageInfo imageInfo{
				.imageView = outImageView,
				.imageLayout = vk::ImageLayout::eGeneral,
			};

			// Uniform buffer
			vk::DescriptorBufferInfo bufferInfo{
				.buffer = uniformBuffers[i],
				.offset = 0,
				.range = sizeof(UniformBufferObject) // vk::WholeSize
			};

			vk::WriteDescriptorSet descriptorWrite[] = {
				{
					.pNext = &accelInfo,
					.dstSet = descriptorSets[i],
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
				},
				{
					.dstSet = descriptorSets[i],
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageImage,
					.pImageInfo = &imageInfo,
				},
				{
					.dstSet = descriptorSets[i],
					.dstBinding = 2,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &bufferInfo,
				},
			};

			//std::array descriptorWrite{
			//	vk::WriteDescriptorSet{
			//		.dstSet = descriptorSets[i],
			//		.dstBinding = 0,
			//		.dstArrayElement = 0,
			//		.descriptorCount = 1,
			//		.descriptorType = vk::DescriptorType::eUniformBuffer,
			//		.pBufferInfo = &bufferInfo,
			//	},
			//	vk::WriteDescriptorSet{
			//		.dstSet = descriptorSets[i],
			//		.dstBinding = 1,
			//		.dstArrayElement = 0,
			//		.descriptorCount = 1,
			//		.descriptorType = vk::DescriptorType::eStorageBuffer,
			//		.pBufferInfo = &storageBufferInfoLastFrame,
			//	},
			//	vk::WriteDescriptorSet{
			//		.dstSet = descriptorSets[i],
			//		.dstBinding = 2,
			//		.dstArrayElement = 0,
			//		.descriptorCount = 1,
			//		.descriptorType = vk::DescriptorType::eStorageBuffer,
			//		.pBufferInfo = &storageBufferInfoCurrentFrame,
			//	},
			//};

			device.updateDescriptorSets(descriptorWrite, {});
		}
	}

	// Command buffers
	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = queueIndex,
		};

		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createCommandBuffers()
	{
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT,
		};

		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
		computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
		rtCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// Semaphores & fences
	void createSyncObjects()
	{
		assert(presentCompleteSemaphores.empty() &&
			renderFinishedSemaphores.empty() &&
			inFlightFences.empty() &&
			computeInFlightFences.empty() &&
			computeFinishedSemaphores.empty());

		for (size_t i = 0; i < swapChainImages.size(); ++i)
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			computeFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(device,
				vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
			computeInFlightFences.emplace_back(device,
				vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
		}
	}

	// Texture images
	void transitionImageLayout(
		const vk::raii::CommandBuffer& commandBuffer,
		const vk::Image& image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::ImageAspectFlags imageAspectMask = vk::ImageAspectFlagBits::eColor,
		uint32_t mipLevels = 1)
	{
		vk::AccessFlags2 srcAccessMask;
		vk::AccessFlags2 dstAccessMask;
		vk::PipelineStageFlags2 srcStageMask;
		vk::PipelineStageFlags2 dstStageMask;

		if (oldLayout == vk::ImageLayout::eUndefined &&
			newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			srcAccessMask = {};
			dstAccessMask = vk::AccessFlagBits2::eTransferWrite;

			srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
			dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
			newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
			dstAccessMask = vk::AccessFlagBits2::eShaderRead;

			srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
			dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
		}
		else if (oldLayout == vk::ImageLayout::eGeneral &&
			newLayout == vk::ImageLayout::eTransferSrcOptimal)
		{
			// (RT) outImage: RT가 storage로 쓴 것 → copy source
			srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
			dstAccessMask = vk::AccessFlagBits2::eTransferRead;

			srcStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
			dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
		}
		else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal &&
			newLayout == vk::ImageLayout::eGeneral)
		{
			// (RT) outImage: copy 끝 → 다음 프레임 RT 쓰기를 위해 general 복원
			srcAccessMask = vk::AccessFlagBits2::eTransferRead;
			dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

			srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
			dstStageMask = vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
			newLayout == vk::ImageLayout::ePresentSrcKHR)
		{
			// (RT) swapchain: copy 받음 → present
			srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
			dstAccessMask = {};

			srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
			dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		transitionImageLayout(commandBuffer, 
			image, 
			oldLayout, newLayout, 
			srcAccessMask, dstAccessMask, 
			srcStageMask, dstStageMask, 
			imageAspectMask);
	}

	// Render target
	void transitionImageLayout(
		const vk::raii::CommandBuffer& commandBuffer,
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2        srcAccessMask,
		vk::AccessFlags2        dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask,
		vk::ImageAspectFlags imageAspectMask,
		uint32_t mipLevels = 1)
	{
		// 이미지 메모리 베리어란? 총 3가지 용도가 있음
		// 1. 동기화를 도와주는 목적, 리소스의 접근을 섬세하게 제어해줄 수 있다 -> accessmask, stagemask
		// 2. 여러 개의 queue family를 쓸 때 (2개의 queue가 동시에 돌아가는 등) queue 동기화
		// 3. image layout 전환, vkBufferMemoryBarrier라는 것도 있는데 비슷하게 생김
		// 그런데 buffermemorybarrier에는 layout 넣어주는 곳이 없음 -> 프로그래머가 알아서하라는 얘기
		// 반면에 image layout은 굉장히 복잡하기 때문에 드라이버가 해줌
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = srcStageMask,
			.srcAccessMask = srcAccessMask,
			.dstStageMask = dstStageMask,
			.dstAccessMask = dstAccessMask,
			.oldLayout = oldLayout,	// 현재 레이아웃
			.newLayout = newLayout,	// 변경할 레이아웃
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,	// queue family 1개만 쓰고 있음
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,		// layout이 변경될 대상 이미지
			.subresourceRange = {
				.aspectMask = imageAspectMask,
				.baseMipLevel = 0,
				.levelCount = mipLevels,
				.baseArrayLayer = 0,
				.layerCount = 1} 
		};

		vk::DependencyInfo dependency_info = {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier };	// imageMemoryBarrier를 쓸 것

		commandBuffer.pipelineBarrier2(dependency_info);
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		const auto& commandBuffer = commandBuffers[frameIndex];
		commandBuffer.reset();
		commandBuffer.begin({});

		//transitionImageLayout(commandBuffer,
		//	swapChainImages[imageIndex],
		//	vk::ImageLayout::eUndefined,
		//	vk::ImageLayout::eColorAttachmentOptimal,
		//	{},
		//	vk::AccessFlagBits2::eColorAttachmentWrite,
		//	vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		//	vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		//	vk::ImageAspectFlagBits::eColor);

		//transitionImageLayout(commandBuffer,
		//	*depthImage,
		//	vk::ImageLayout::eUndefined,
		//	vk::ImageLayout::eDepthAttachmentOptimal,
		//	vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		//	vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		//	vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		//	vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		//	vk::ImageAspectFlagBits::eDepth);

		vk::ClearValue clearColor = { .color = {.float32 = {{ 0.0f, 0.0f, 0.0f, 1.0f }} } };
		vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = swapChainImageViews[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor,
		};

		vk::ClearValue clearDepth = { .depthStencil = {.depth = 1.0f, .stencil = 0 } };
		vk::RenderingAttachmentInfo depthAttachmentInfo = {
			.imageView = depthImageView,
			.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eDontCare,
			.clearValue = clearDepth,
		};

		vk::RenderingInfo renderingInfo = {
			.renderArea = {.offset = {0,0}, .extent = swapchainExtent},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo,
			.pDepthAttachment = &depthAttachmentInfo,
		};

		commandBuffer.beginRendering(renderingInfo);
		{
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

			//commandBuffer.bindVertexBuffers(0, *vertexBuffer, { 0 });
			commandBuffer.bindVertexBuffers(0, *shaderStorageBuffers[frameIndex], { 0 });
			//commandBuffer.bindIndexBuffer(*indexBuffer, 0, 
			//										   vk::IndexTypeValue<decltype(indices)::value_type>::value);

			commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
													  static_cast<float>(swapchainExtent.width),
													  static_cast<float>(swapchainExtent.height),
													  0.0f, 1.0f));
			commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

			//commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, 
			//											  graphicsPipelineLayout, 0, 
			//											  *descriptorSets[frameIndex], nullptr);

			//commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
		}
		commandBuffer.endRendering();

		//transitionImageLayout(commandBuffer,
		//	swapChainImages[imageIndex],
		//	vk::ImageLayout::eColorAttachmentOptimal,
		//	vk::ImageLayout::ePresentSrcKHR,
		//	vk::AccessFlagBits2::eColorAttachmentWrite,
		//	{},
		//	vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		//	vk::PipelineStageFlagBits2::eBottomOfPipe,
		//	vk::ImageAspectFlagBits::eColor);

		commandBuffer.end();
	}

	void recordComputeCommandBuffer()
	{
		const auto& commandBuffer = computeCommandBuffers[frameIndex];
		commandBuffer.reset();
		commandBuffer.begin({});
		// Compute pipeline
		{
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *comptePipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *descriptorSets[frameIndex], {});
			commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);
		}
		commandBuffer.end();
	}

	void recordRTCommandBuffer(const uint32_t imageIndex)
	{
		static const vk::ImageCopy copyRegion = {
			.srcSubresource = {
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.dstSubresource = {
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.extent = {WIDTH, HEIGHT, 1},
		};

		const auto& commandbuffer = rtCommandBuffers[frameIndex];
		commandbuffer.reset();
		commandbuffer.begin({});
		// RT pipeline
		{
			commandbuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rtPipeline);
			commandbuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, rtPipelineLayout, 0, *descriptorSets[frameIndex], {});
			
			// 하나의 TLAS로 구성을 해서 drawcall(traceRays)를 한번만 호출을 한다
			// graphics pipeline에서는 매쉬들 개수만큼 draw call을 하는데 (여러가지 최적화 빼고 생각하면)
			// raytracing은 그렇지 않다
			// 그래서 거대한 scene에 대해서 대량의 데이터가 한번의 draw call에 다 참조가 되어야한다
			commandbuffer.traceRaysKHR(rgenSbt, missSbt, hitgSbt, {}, WIDTH, HEIGHT, 1);
			// 이 command가 제출이 되고 실행이 되면 raygen 실행
			// ray가 교차됨 -> chit 호출 -> material마다 chit가 다 다름
			// 교차한 순간에 어떤 material인지 알고 그에 해당하는 chit를 호출을 해야함
			// 그게 자동으로 일어나게하기 위해서 geometry record맞게 쉐이더 핸들이 저장되어 있어야하고
			// 그에 맞는 chit가 호출이 되어야함
			// geometry가 100가 있다면 100개의 record가 있을 텐데
			// 그 중에 어떤 record인지를 어떻게 아는가? 어떻게 찾아가는가? -> indexing rule
			// rgen : sbt.deviceaddress에 handle이 저장되어 있어야함
			// miss : sbt.deviceAddress + sbt.stride * missIndex -> 그러니까 miss shader handle은 연달아서 있어야함 -> traceRayEXT에 missIndex 넣어주는 부분 있음
			// chit : sbt.deviceAddress + sbt.stride(record size) * (instanceShaderBindingTableRecordOffset + geometryIndex * sbtRecordStride + sbtRecordOffset)
			// sbtRecordStride, sbtRecordOffset -> traceRayEXT에 들어가 있음
			// geometryIndex = gl_GeometryIndexEXT, BLAS 내 geometry index
			// instanceShaderBindingTableRecordOffset -> TLAS 만들때 지정
			// 이렇게 계산되어야 instance마다, 그리고 geometry마다 구분이 됨
			// SBT -> 이렇게 handle을 찾아가는 것 뿐만 아니라 추가적인 customData도 필요
			// ray는 한번만 쏴주고 각 object마다의 data도 필요하기 때문에 이러한 구조가 생김

			transitionImageLayout(
				commandbuffer, 
				outImage, 
				vk::ImageLayout::eGeneral, 
				vk::ImageLayout::eTransferSrcOptimal);

			transitionImageLayout(
				commandbuffer,
				swapChainImages[imageIndex],
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal);

			commandbuffer.copyImage(
				outImage, vk::ImageLayout::eTransferSrcOptimal, 
				swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, 
				copyRegion);

			transitionImageLayout(
				commandbuffer,
				outImage,
				vk::ImageLayout::eTransferSrcOptimal,
				vk::ImageLayout::eGeneral);

			transitionImageLayout(
				commandbuffer,
				swapChainImages[imageIndex],
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::ePresentSrcKHR);
		}
		commandbuffer.end();
	}

	void updateUniformBuffer(uint32_t currentFrame)
	{
		//static auto startTime = std::chrono::high_resolution_clock::now();

		//auto currentTime = std::chrono::high_resolution_clock::now();
		//float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		//UniformBufferObject ubo{};
		//ubo.model = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));
		//ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		//ubo.proj = glm::perspective(glm::radians(45.0f),
		//							static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height),
		//							0.1f, 10.0f);
		//ubo.proj[1][1] *= -1;

		//memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));

		//UniformBufferObject ubo{};
		//ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

		//memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));

		UniformBufferObject ubo{ .cameraPos = {0,0,10}, .yFov_degree = 60 };
		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void drawFrame()
	{
		//// Submit computeQueue
		//{
		//	auto fenceResult = device.waitForFences(*computeInFlightFences[frameIndex], vk::True, UINT64_MAX);
		//	if (fenceResult != vk::Result::eSuccess) {
		//		throw std::runtime_error("failed to wait for fence!");
		//	}
		//	device.resetFences(*computeInFlightFences[frameIndex]);
		//	computeCommandBuffers[frameIndex].reset();
		//	recordComputeCommandBuffer();

		//	const vk::SubmitInfo submitInfo{
		//		.commandBufferCount = 1,
		//		.pCommandBuffers = &*computeCommandBuffers[frameIndex],
		//		.signalSemaphoreCount = 1,
		//		.pSignalSemaphores = &*computeFinishedSemaphores[frameIndex],
		//	};
		//	computeQueue.submit(submitInfo, *computeInFlightFences[frameIndex]);
		//}

		// Submit graphicsQueue & present
		{
			auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
			if (fenceResult != vk::Result::eSuccess) {
				throw std::runtime_error("failed to wait for fence!");
			}
			device.resetFences(*inFlightFences[frameIndex]);

			updateUniformBuffer(frameIndex);
			auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
			recordRTCommandBuffer(imageIndex);

			vk::Semaphore waitSemaphores[] = { presentCompleteSemaphores[frameIndex] };
			vk::PipelineStageFlags waitDestinationStageMask[] = { vk::PipelineStageFlagBits::eTransfer };

			const vk::SubmitInfo submitInfo{
				.waitSemaphoreCount = sizeof(waitSemaphores) / sizeof(waitSemaphores[0]),
				.pWaitSemaphores = waitSemaphores,
				.pWaitDstStageMask = waitDestinationStageMask,
				.commandBufferCount = 1,
				.pCommandBuffers = &*rtCommandBuffers[frameIndex],
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex],
			};

			graphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

			const vk::PresentInfoKHR presentInfoKHR{
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
				.swapchainCount = 1,
				.pSwapchains = &*swapChain,
				.pImageIndices = &imageIndex
			};

			result = graphicsQueue.presentKHR(presentInfoKHR);
		}

		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();

			double currentTime = glfwGetTime();
			lastFrameTime = (currentTime - lastTime) * 1000.0;
			lastTime = currentTime;
		}

		device.waitIdle();
	}

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}