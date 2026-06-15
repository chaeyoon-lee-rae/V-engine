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

const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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

struct UniformBufferObject {
	float deltaTime = 1.0f;
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
		createSwapchainImageViews();
		createDescriptorSetLayout();
		createComputeDescriptorSetLayout();
		createGraphicsPipeline();
		createComputePipeline();
		// createRTPipeline();
		createCommandPool();
		createDepthResources();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		//loadModel();
		//createVertexBuffer();
		//createIndexBuffer();
		createShaderStorageBuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
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
	std::vector<const char*> requiredDeviceExtension = { vk::KHRSwapchainExtensionName };

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
		bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
			return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtension](auto const& availableDeviceExtension) {
				return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
				});
			});

		// Device features
		auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, 
														     vk::PhysicalDeviceVulkan13Features, 
															 vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
															 vk::PhysicalDeviceVulkan11Features>();
		bool supportsRequiredFeatures = 
			features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
			features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
			features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters;

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
		vk::StructureChain<vk::PhysicalDeviceFeatures2,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
			vk::PhysicalDeviceVulkan11Features>
			featureChain = {
				{.features = {.samplerAnisotropy = true}},
				{.synchronization2 = true, .dynamicRendering = true},
				{.extendedDynamicState = true },
				{.shaderDrawParameters = true }, 
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
			.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
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
										vk::ImageAspectFlags aspectFlags, 
										uint32_t mipLevels)
	{
		vk::ImageViewCreateInfo viewInfo{
			.image = image,
			.viewType = vk::ImageViewType::e2D,
			.format = format,
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

	void createSwapchainImageViews()
	{
		assert(swapChainImageViews.empty());

		swapChainImageViews.reserve(swapChainImages.size());
		for (auto& image : swapChainImages) {
			swapChainImageViews.emplace_back(
				createImageView(image, 
								swapChainSurfaceFormat.format, 
								vk::ImageAspectFlagBits::eColor, 1)
			);
		}
	}

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
		depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
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

	// Texture images
	void transitionImageLayout(
		vk::raii::CommandBuffer &commandBuffer,
		const vk::raii::Image &image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		uint32_t mipLevels)
	{
		vk::ImageMemoryBarrier barrier{
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image = image,
			.subresourceRange = {
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.levelCount = mipLevels,
				.layerCount = 1
			}
		};

		vk::PipelineStageFlags sourceStage;
		vk::PipelineStageFlags destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined && 
			newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			barrier.srcAccessMask = {};
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && 
				 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
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
							  mipLevels);
		copyBufferToImage(commandBuffer, stagingBuffer, textureImage, texWidth, texHeight);
		endSingleTimeCommands(std::move(commandBuffer));

		generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(*textureImage, 
										   vk::Format::eR8G8B8A8Srgb, 
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

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) 
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

		std::array poolSize{
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)
		};

		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
			.pPoolSizes = poolSize.data(),
		};

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	void createDescriptorSets()
	{
		//std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);

		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data(),
		};

		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vk::DescriptorBufferInfo bufferInfo{
				.buffer = uniformBuffers[i],
				.offset = 0,
				.range = sizeof(UniformBufferObject) // vk::WholeSize
			};

			//vk::DescriptorImageInfo imageInfo{
			//	.sampler = textureSampler,
			//	.imageView = textureImageView,
			//	.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			//};

			vk::DescriptorBufferInfo storageBufferInfoLastFrame(
				shaderStorageBuffers[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
			vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
				shaderStorageBuffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);

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

			std::array descriptorWrite{
				vk::WriteDescriptorSet{
					.dstSet = descriptorSets[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &bufferInfo,
				},
				vk::WriteDescriptorSet{
					.dstSet = descriptorSets[i],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &storageBufferInfoLastFrame,
				},
				vk::WriteDescriptorSet{
					.dstSet = descriptorSets[i],
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &storageBufferInfoCurrentFrame,
				},
			};

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

	void transition_image_layout(vk::Image               image,
								 vk::ImageLayout         old_layout,
								 vk::ImageLayout         new_layout,
								 vk::AccessFlags2        src_access_mask,
								 vk::AccessFlags2        dst_access_mask,
								 vk::PipelineStageFlags2 src_stage_mask,
								 vk::PipelineStageFlags2 dst_stage_mask,
								 vk::ImageAspectFlags	image_aspect_flags)
	{
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = src_stage_mask,
			.srcAccessMask = src_access_mask,
			.dstStageMask = dst_stage_mask,
			.dstAccessMask = dst_access_mask,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				   .aspectMask = image_aspect_flags,
				   .baseMipLevel = 0,
				   .levelCount = 1,
				   .baseArrayLayer = 0,
				   .layerCount = 1} };

		vk::DependencyInfo dependency_info = {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier };

		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		const auto& commandBuffer = commandBuffers[frameIndex];
		commandBuffer.reset();
		commandBuffer.begin({});

		transition_image_layout(swapChainImages[imageIndex],
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::ImageAspectFlagBits::eColor);

		transition_image_layout(*depthImage,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eDepthAttachmentOptimal,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::ImageAspectFlagBits::eDepth);

		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = swapChainImageViews[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor,
		};

		vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
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

		transition_image_layout(swapChainImages[imageIndex],
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			{},
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eBottomOfPipe,
			vk::ImageAspectFlagBits::eColor);

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

		UniformBufferObject ubo{};
		ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void drawFrame()
	{
		// Submit computeQueue
		{
			auto fenceResult = device.waitForFences(*computeInFlightFences[frameIndex], vk::True, UINT64_MAX);
			if (fenceResult != vk::Result::eSuccess) {
				throw std::runtime_error("failed to wait for fence!");
			}
			updateUniformBuffer(frameIndex);
			device.resetFences(*computeInFlightFences[frameIndex]);
			computeCommandBuffers[frameIndex].reset();
			recordComputeCommandBuffer();

			const vk::SubmitInfo submitInfo{
				.commandBufferCount = 1,
				.pCommandBuffers = &*computeCommandBuffers[frameIndex],
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &*computeFinishedSemaphores[frameIndex],
			};
			computeQueue.submit(submitInfo, *computeInFlightFences[frameIndex]);
		}

		// Submit graphicsQueue & present
		{
			auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
			if (fenceResult != vk::Result::eSuccess) {
				throw std::runtime_error("failed to wait for fence!");
			}
			device.resetFences(*inFlightFences[frameIndex]);

			auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
			recordCommandBuffer(imageIndex);

			vk::Semaphore waitSemaphores[] = { presentCompleteSemaphores[frameIndex], computeFinishedSemaphores[frameIndex] };
			vk::PipelineStageFlags waitDestinationStageMask[] = { vk::PipelineStageFlagBits::eVertexInput,
																  vk::PipelineStageFlagBits::eColorAttachmentOutput };

			const vk::SubmitInfo submitInfo{
				.waitSemaphoreCount = 2,
				.pWaitSemaphores = waitSemaphores,
				.pWaitDstStageMask = waitDestinationStageMask,
				.commandBufferCount = 1,
				.pCommandBuffers = &*commandBuffers[frameIndex],
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