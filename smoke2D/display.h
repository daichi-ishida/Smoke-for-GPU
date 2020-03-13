#pragma once
#define GLFW_INCLUDE_GLU
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>

struct DestroyglfwWin
{
    void operator()(GLFWwindow* ptr)
    {
         glfwDestroyWindow(ptr);
    }
};

class Display
{
public:
    Display(){}
    ~Display(){}

    void createWindow();
    void updateWindow();
    bool isClosing() const;
    void close() const;

private:
    std::unique_ptr<GLFWwindow, DestroyglfwWin> m_window;
};

