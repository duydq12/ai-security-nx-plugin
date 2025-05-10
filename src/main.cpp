#include <iostream>
#include <vector>
#include <string>

#include "device_agent.h"
#include <filesystem>
#include <opencv2/highgui/highgui.hpp>
#include <nx/sdk/analytics/helpers/consuming_device_agent.h>
#include <nx/sdk/ptr.h>
#include <nx/sdk/helpers/string.h>
#include <nx/sdk/analytics/i_uncompressed_video_frame.h>


using namespace nx::sdk;
using namespace nx::sdk::analytics;


class TestVideoFrame : public RefCountable<IUncompressedVideoFrame> {
public:
    TestVideoFrame(const cv::Mat &rgbImage)
            : m_image(rgbImage),
              m_timestampUs(std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count()),
              m_refCount(1) {}

    virtual int width() const override { return m_image.cols; }

    virtual int height() const override { return m_image.rows; }

    virtual int planeCount() const override { return 1; }

    virtual int dataSize(int plane) const override {
        if (plane != 0)
            return 0;
        return m_image.total() * m_image.elemSize();
    }


    PixelFormat pixelFormat() const override { return PixelFormat::rgb; }

    void getPixelAspectRatio(PixelAspectRatio *outValue) const override {
        outValue->numerator = 1;
        outValue->denominator = 1;
    }

    nx::sdk::IList<nx::sdk::analytics::IMetadataPacket> *getMetadataList() const override {
        return nullptr; // No metadata in test frame
    }

    virtual int lineSize(int plane) const override { return m_image.step; }

    virtual const char *data(int plane) const override { return reinterpret_cast<const char *>(m_image.data); }

    virtual int64_t timestampUs() const override { return m_timestampUs; }

    virtual int addRef() const override { return ++m_refCount; }

    virtual int releaseRef() const override {
        if (--m_refCount <= 0)
            delete this;
        return m_refCount;
    }

private:
    int64_t m_timestampUs;
    mutable std::atomic<int> m_refCount;
    cv::Mat m_image;
};

class MockDeviceInfo : public RefCountable<nx::sdk::IDeviceInfo> {
public:
    MockDeviceInfo(
            const std::string &id,
            const std::string &vendor,
            const std::string &model
    )
            :
            m_id(id),
            m_vendor(vendor),
            m_model(model),
            m_refCount(1) {
    }

    // IDeviceInfo interface implementation
    const char *id() const override { return m_id.c_str(); }

    const char *vendor() const override { return m_vendor.c_str(); }

    const char *model() const override { return m_model.c_str(); }

    const char *firmware() const override { return "1.0"; }

    const char *name() const override { return "test"; }

    const char *url() const override { return "test"; }

    const char *login() const override { return "test"; }

    const char *password() const override { return "test"; }

    const char *sharedId() const override { return "test"; }

    const char *logicalId() const override { return "test"; }

    int channelNumber() const override { return 0; }

    int addRef() const override { return ++m_refCount; }

    int releaseRef() const override {
        if (--m_refCount <= 0)
            delete this;
        return m_refCount;
    }

private:
    std::string m_id;
    std::string m_vendor;
    std::string m_model;
    mutable std::atomic<int> m_refCount;
};


int main() {
    std::filesystem::path pluginHomeDir("/home/developer/resources");
    nx::sdk::Ptr<nx::sdk::IDeviceInfo> deviceInfo(
            new MockDeviceInfo("mock_device_001", "MockVendor", "VirtualCamera_Model_X"));
    const nx::sdk::IDeviceInfo *deviceInfoInterface = deviceInfo.get();
    nx_meta_plugin::DeviceAgent deviceAgent(deviceInfoInterface, pluginHomeDir);
    nx::sdk::Result<void> *outValue;
    const nx::sdk::analytics::IMetadataTypes *neededMetadataTypes;
    deviceAgent.doSetNeededMetadataTypes(outValue, neededMetadataTypes);

/// Test with image
    std::string imagePath = "/home/developer/resources/congan.jpg"; // CHANGE THIS
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) throw std::runtime_error("Failed to load image");
//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    auto testFrame = new TestVideoFrame(img);  // Uses RefCountable
    deviceAgent.processFrame(testFrame);

    cv::imwrite("Frame.jpg", img);

//    /// Test with video
//    cv::VideoCapture cap("/home/developer/resources/tramgiam-cong-an-pham-nhan copy.mp4");;
//    if (!cap.isOpened()) {
//        std::cout << "Error opening video stream or file" << std::endl;
//        return -1;
//    }
//    while (1) {
//        cv::Mat frame;
//        cap >> frame;
//
//        // If the frame is empty, break immediately
//        if (frame.empty())
//            break;
//
//        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
//
//        auto testFrame = new TestVideoFrame(frame);  // Uses RefCountable
//        deviceAgent.processFrame(testFrame);
//
//        // Display the resulting frame
//        cv::imshow("Frame", frame);
//
//        // Press  ESC on keyboard to exit
//        char c = (char) cv::waitKey(25);
//        if (c == 27)
//            break;
//    }
//
//    // When everything done, release the video capture object
//    cap.release();
//
//    // Closes all the frames
//    cv::destroyAllWindows();

    return 0;
}