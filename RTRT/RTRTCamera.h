#pragma once

#include "Engine.h"
#include <cstdio>
#include "Essential.h"
#include "Vector.h"
#include "Input.h"
#include <GLFW/glfw3.h>


struct CameraInfo
{
	CameraInfo(void)
	{
	}

	CameraInfo(const Vector &position, float yaw, float pitch)
		: position(position), yaw(yaw), pitch(pitch)
	{
	}

	CUDA_HOST_DEVICE Vector getDirection() const
	{
		return Vector(
			cos(pitch) * sin(yaw), 
			sin(-pitch),
			cos(pitch) * cos(yaw)
		);
	}

	CUDA_HOST_DEVICE Vector getRight() const
	{
		return Vector(
			sin(yaw - efl::PI/2.0f), 
			0,
			cos(yaw - efl::PI/2.0f)
			);
	}

	CUDA_HOST_DEVICE Vector getUp() const
	{
		return Vector::crossProduct(getRight(), getDirection());
	}

	float yaw, pitch;
	Vector position;
};


struct RTRTCamera
{
	RTRTCamera(int width, int height)
		: cam(Vector(0.0f, 0.0f, 0.0f), 0.0f, 0.0f), width(width), height(height), dir(0.0f)
	{
		springiness = 55.0f;
		mouseSpeed = 0.0009f;
		cameraSpeed = 5.0f;
		usex = width;
		usey = height;
		glfwSetInputMode(Engine::getWindow(), GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

	void keyPress(int key, int scancode, int action, int mods) 
	{
		currentMod = mods;

		if(action == GLFW_REPEAT)
			return;

		if(currentMod == GLFW_MOD_SHIFT && action == GLFW_PRESS)
			glfwSetInputMode(Engine::getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		else
			glfwSetInputMode(Engine::getWindow(), GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

		float t = (action == GLFW_PRESS? 1.0f : -1.0f);
		switch(key)
		{
		case GLFW_KEY_W:
			translate(Vector(0, 0, -1.) * t);
			break;
		case GLFW_KEY_A:
			translate(Vector(1., 0, 0) * t);
			break;
		case GLFW_KEY_S:
			translate(Vector(0, 0, 1.) * t);
			break;
		case GLFW_KEY_D:
			translate(Vector(-1., 0, 0) * t);
			break;
		case GLFW_KEY_Q:
			translate(Vector(0, 1., 0) * t);
			break;
		case GLFW_KEY_E:
			translate(Vector(0, -1., 0) * t);
			break;
		default:
			break;
		}
	}

	void translate(const Vector &_dir)
	{
		dir += _dir;
	}

	void mouseMove(double x, double y) 
	{
		static float lastTime = glfwGetTime();
		float currentTime = glfwGetTime();
		float dt = currentTime - lastTime;
		lastTime = currentTime;

		float d = 1 - exp(-0.693147f * springiness * dt);//log(0.5) = -0.693147
		usex += (x - usex) * d;
		usey += (y - usey) * d;

		if(currentMod != GLFW_MOD_SHIFT)
		{
			rotate(float(width / 2 - x) * mouseSpeed, float(height / 2 - y) * mouseSpeed);
			Input::setMousePosition(width / 2, height / 2);
		}
	}

	void update(float dt)
	{
		float distance = dt * cameraSpeed;
		Vector direction = cam.getDirection();
		Vector right = cam.getRight();
		Vector up = Vector::crossProduct(right, direction);
		cam.position += (direction * dir.z + up * dir.y + right * dir.x) * distance;
	}

	void rotate(float _yaw, float _pitch)
	{
		cam.pitch += _pitch;
		cam.yaw += _yaw;
		if(cam.pitch > efl::PI / 2.)
			cam.pitch = efl::PI / 2.;
		if(cam.pitch < -efl::PI / 2.)
			cam.pitch = -efl::PI / 2.;
	}

	CameraInfo cam;

	Vector dir;
	float cameraSpeed;
	float mouseSpeed;
	float springiness;
	float usex, usey;
	int width, height;
	int currentMod;
};