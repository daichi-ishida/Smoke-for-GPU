#include "display.h"
#include "Smoke.h"
#include "renderer.h"
#include "simulator.h"
#include "constants.h"

#include <cuda_runtime.h>
#include "helper_cuda.h"

int main()
{
    findCudaDevice(0, nullptr);

    int step = 1; // simulation step counter
    int frame = 1; // rendered frame counter

    // Display display;
    // display.createWindow();

    std::unique_ptr<Smoke> data{ std::make_unique<Smoke>() };
    std::unique_ptr<Simulator> simulator{ std::make_unique<Simulator>(data) };
    std::unique_ptr<Renderer> renderer{ std::make_unique<Renderer>(data) };

    data->initialize();
    renderer->initialize();

    renderer->render();
    renderer->saveImage();
    data->setNextShutterTime();

    printf("\n*** SIMULATION START ***\n");

    while (frame <= END_FRAME)
    {
        printf("\n=== STEP %d : RENDERED %d ===\n", step, frame);

        // simulation
        simulator->update();

        // rendering
        renderer->render();
        if (data->isTimeToRender)
        {
            if (SAVE_IMAGE)
            {
                renderer->saveImage();
            }
            if (SAVE_VDB)
            {
                data->saveVDB();
            }
            data->isTimeToRender = false;
            ++frame;
        }

        ++step;
        // display.updateWindow();
    }
    printf("\n*** SIMULATION END ***\n");

    // display.close();

    renderer.reset();
    data.reset();

    checkCudaErrors(cudaDeviceReset());
}