#pragma once

#include <memory>

struct GLFWwindow;

namespace pb {

class PreviewViewer {
public:
    PreviewViewer(GLFWwindow* window, int face_resolution);
    ~PreviewViewer();

    PreviewViewer(const PreviewViewer&) = delete;
    PreviewViewer& operator=(const PreviewViewer&) = delete;

    void update_input();
    void render(int fb_width, int fb_height);
    void draw_ui();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
