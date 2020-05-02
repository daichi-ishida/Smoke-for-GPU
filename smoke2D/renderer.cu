#include "renderer.h"
#include "constants.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>

Renderer::Renderer(std::unique_ptr<Smoke>& data) :m_data(data)
{
}

void Renderer::saveImage()
{
    static int count = 0;
    char filename[1024];

    snprintf(filename, sizeof(filename), "img/res%dx%d_%04d.png", xRes, yRes, count++);

    h_image = d_image;

    unsigned char* data = reinterpret_cast<unsigned char*>(h_image.data()); // 3 components (R, G, B)

    int saved = stbi_write_png(filename, WIN_WIDTH, WIN_HEIGHT, 3, data, 0);
}
