#pragma once

#include <memory>

struct GLFWwindow;

namespace pb {

struct PlanetState;
class  ParamRegistry;
class  Pipeline;
class  ProgressSink;

class PreviewViewer {
public:
    PreviewViewer(GLFWwindow* window,
                  PlanetState& state,
                  ParamRegistry& registry,
                  Pipeline& pipeline);
    ~PreviewViewer();

    PreviewViewer(const PreviewViewer&)            = delete;
    PreviewViewer& operator=(const PreviewViewer&) = delete;

    //ProgressSink fed to pipeline.run() from main(). Owned by the viewer so
    //the panel can render live stage + fraction.
    ProgressSink& progress_sink();

    void update_input();
    void render(int fb_width, int fb_height);
    void draw_ui();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
