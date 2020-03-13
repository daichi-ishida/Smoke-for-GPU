#include "renderer.h"
#include "constants.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>

Renderer::Renderer(std::unique_ptr<Smoke>& data) : pbo(0), tex_buffer(0), cuda_pbo_resource(nullptr), m_data(data)
{
    stbi_flip_vertically_on_write(1);
}

void Renderer::saveImage()
{
    static int count = 0;
    char filename[1024];

    snprintf(filename, sizeof(filename), "img/res%dx%d_%04d.png", xRes, yRes, count++);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    int x = viewport[0];
    int y = viewport[1];
    int width = viewport[2];
    int height = viewport[3];

    unsigned char* data = new unsigned char[width * height * 3]; // 3 components (R, G, B)

    if (!data)
        return;

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

    int saved = stbi_write_png(filename, width, height, 3, data, 0);

    delete[] data;
}
