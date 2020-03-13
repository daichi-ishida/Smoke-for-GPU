#include "display.h"
#include "constants.h"

void Display::createWindow()
{
    if (glfwInit() == GLFW_FALSE)
    {
        fprintf(stderr, "Initialization failed!\n");
    }

    if (OFFSCREEN_MODE)
    {
        glfwWindowHint(GLFW_VISIBLE, 0);
    }

    m_window.reset(glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE, NULL, NULL));

    if (m_window == nullptr)
    {
        fprintf(stderr, "Window creation failed!");
        glfwTerminate();
    }
    glfwMakeContextCurrent(m_window.get());

    glewExperimental = true;
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "GLEW initialization failed!\n");
    }
}

void Display::updateWindow()
{
    glfwSwapBuffers(m_window.get());
    glfwPollEvents();
}

bool Display::isClosing() const
{
    return glfwWindowShouldClose(m_window.get()) || glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE);
}

void Display::close() const
{
    glfwTerminate();
}