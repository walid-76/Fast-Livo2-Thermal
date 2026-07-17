#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

class ThermalConverter : public rclcpp::Node
{
public:
    ThermalConverter() : Node("thermal_converter"),
        p_low_smooth_(0.0f), p_high_smooth_(65535.0f),
        initialized_(false)
    {
        // Declare parameters (tunable at runtime via ros2 param set)
        this->declare_parameter("alpha",          0.05);  // EMA smoothing (lower = more stable)
        this->declare_parameter("low_percentile",  2.0);  // cut bottom N%
        this->declare_parameter("high_percentile",98.0);  // cut top N%
        this->declare_parameter("clahe_clip",      2.0);  // lower = less noise amp
        this->declare_parameter("clahe_grid",      8);    // tile size
        // Colormap options (cv::ColormapTypes):
        //   COLORMAP_INFERNO   = 8  (dark→purple→orange→yellow) — best for thermal
        //   COLORMAP_JET       = 2  (blue→green→red)
        //   COLORMAP_HOT       = 11 (black→red→yellow→white)
        //   COLORMAP_MAGMA     = 9  (dark→purple→pink→white)
        //   COLORMAP_RAINBOW   = 4
        this->declare_parameter("colormap", 8);           // default: INFERNO

        sub_ = image_transport::create_subscription(
            this, "/image_raw",
            std::bind(&ThermalConverter::imageCallback, this, std::placeholders::_1),
            "raw");

        // Publisher 1: mono8 — for FAST-LIVO2 / any algorithm
        pub_mono_  = image_transport::create_publisher(this, "/thermal/stabilized_8bit");

        // Publisher 2: rgb8 colorized — for RViz visualization and debugging
        pub_color_ = image_transport::create_publisher(this, "/thermal/colorized");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try {
            // ── 1. Decode RAW16 ──────────────────────────────────────────
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "mono16");
            cv::Mat raw16 = cv_ptr->image;

            // ── 2. Compute histogram ─────────────────────────────────────
            const int    histSize  = 65536;
            float        range[]   = {0.f, 65536.f};
            const float *histRange = range;
            cv::Mat hist;
            cv::calcHist(&raw16, 1, nullptr, cv::Mat(),
                         hist, 1, &histSize, &histRange);

            // ── 3. Find per-frame percentile bounds ──────────────────────
            double low_pct  = this->get_parameter("low_percentile").as_double();
            double high_pct = this->get_parameter("high_percentile").as_double();
            float totalPx   = static_cast<float>(raw16.rows * raw16.cols);

            float cumulative = 0.f;
            int p_low_frame  = 0;
            int p_high_frame = 65535;
            bool low_found   = false;

            for (int i = 0; i < histSize; ++i) {
                cumulative += hist.at<float>(i);
                if (!low_found && cumulative >= totalPx * (low_pct / 100.0f)) {
                    p_low_frame = i;
                    low_found   = true;
                }
                if (cumulative >= totalPx * (high_pct / 100.0f)) {
                    p_high_frame = i;
                    break;
                }
            }

            // ── 4. Temporal EMA smoothing of bounds ──────────────────────
            //    Prevents brightness jumps when scene content changes.
            //    alpha=0.05 → ~20-frame time constant at 10 Hz ≈ 2 seconds
            double alpha = this->get_parameter("alpha").as_double();
            if (!initialized_) {
                p_low_smooth_  = static_cast<float>(p_low_frame);
                p_high_smooth_ = static_cast<float>(p_high_frame);
                initialized_   = true;
            } else {
                p_low_smooth_  = alpha * p_low_frame  + (1.0 - alpha) * p_low_smooth_;
                p_high_smooth_ = alpha * p_high_frame + (1.0 - alpha) * p_high_smooth_;
            }

            int p1  = static_cast<int>(p_low_smooth_);
            int p99 = static_cast<int>(p_high_smooth_);
            if (p99 <= p1) p99 = p1 + 1;   // safety guard

            // ── 5. Linear stretch to 8-bit using smoothed bounds ─────────
            double scale = 255.0 / (p99 - p1);
            cv::Mat image8;
            raw16.convertTo(image8, CV_8U, scale, -p1 * scale);

            // ── 6. Optional: light denoise before CLAHE ──────────────────
            //    Removes sensor noise so CLAHE doesn't amplify it.
            cv::GaussianBlur(image8, image8, cv::Size(3, 3), 0.5);

            // ── 7. CLAHE ─────────────────────────────────────────────────
            double clahe_clip = this->get_parameter("clahe_clip").as_double();
            int    clahe_grid = this->get_parameter("clahe_grid").as_int();
            auto   clahe      = cv::createCLAHE(clahe_clip,
                                                cv::Size(clahe_grid, clahe_grid));
            clahe->apply(image8, image8);

            // ── 8. Publish mono8 (for FAST-LIVO2) ───────────────────────
            auto mono_msg = cv_bridge::CvImage(msg->header, "mono8", image8).toImageMsg();
            pub_mono_.publish(mono_msg);

            // ── 9. Apply colormap and publish colorized (for RViz) ───────
            //    Only runs if someone is subscribed — saves CPU when not needed
            if (pub_color_.getNumSubscribers() > 0) {
                int colormap_id = this->get_parameter("colormap").as_int();
                cv::Mat colorized;
                cv::applyColorMap(image8, colorized, colormap_id);

                // applyColorMap outputs BGR — convert to RGB for ROS
                cv::Mat colorized_rgb;
                cv::cvtColor(colorized, colorized_rgb, cv::COLOR_BGR2RGB);

                auto color_msg = cv_bridge::CvImage(msg->header, "rgb8", colorized_rgb).toImageMsg();
                pub_color_.publish(color_msg);
            }

        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    image_transport::Subscriber sub_;
    image_transport::Publisher  pub_mono_;   // /thermal/stabilized_8bit  (mono8)
    image_transport::Publisher  pub_color_;  // /thermal/colorized         (rgb8)

    float p_low_smooth_;
    float p_high_smooth_;
    bool  initialized_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ThermalConverter>());
    rclcpp::shutdown();
    return 0;
}