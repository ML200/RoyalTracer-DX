/******************************************************************************
 * Copyright 1986, 2017 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#pragma once

#include "./glm/glm.hpp"

namespace nv_helpers_dx12
{
//--------------------------------------------------------------------------------------------------
// This is a camera manipulator help class
// It allow to simply do
// - Orbit        (LMB)
// - Pan          (LMB + CTRL  | MMB)
// - Dolly        (LMB + SHIFT | RMB)
// - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
// - Trackball
//
// In a various ways:
// - examiner(orbit around object)
// - walk (look up or down but stays on a plane)
// - fly ( go toward the interest point)
//
// Do use the camera manipulator, you need to do the following
// - Call setWindowSize() at creation of the application and when the window size change
// - Call setLookat() at creation to initialize the camera look position
// - Call setMousePosition() on application mouse down
// - Call mouseMove() on application mouse move
//
// Retrieve the camera matrix by calling getMatrix()
//

class Manipulator
{
public:
  // clang-format off
    enum Modes { Examine, Fly, Walk, Trackball };
    enum Actions { None, Orbit, Dolly, Pan, LookAround };
    struct Inputs {bool lmb=false; bool mmb=false; bool rmb=false; 
                   bool shift=false; bool ctrl=false; bool alt=false;};
  // clang-format on

  /// Main function to call from the application
  /// On application mouse move, call this function with the current mouse position, mouse
  /// button presses and keyboard modifier. The camera matrix will be updated and
  /// can be retrieved calling getMatrix
  Actions mouseMove(int x, int y, const Inputs& inputs);

  /// Set the camera to look at the interest point
  void setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);

  /// Changing the size of the window. To call when the size of the window change.
  /// This allows to do nicer movement according to the window size.
  void setWindowSize(int w, int h);

  /// Setting the current mouse position, to call on mouse button down
  void setMousePosition(int x, int y);

  // Factory.
  static Manipulator& Singleton()
  {
    static Manipulator manipulator;
    return manipulator;
  }

  /// Retrieve the position, interest and up vector of the camera
  void getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const;

  /// Set the manipulator mode, from Examiner, to walk, to fly, ...
  void setMode(Modes mode);

  /// Retrieve the current manipulator mode
  Modes getMode() const;

  /// Setting the roll (radian) around the Z axis
  void setRoll(float roll);

  /// Retrieve the camera roll
  float getRoll() const;

  /// Retrieving the transformation matrix of the camera
  const glm::mat4& getMatrix() const;

  /// Changing the default speed movement
  void setSpeed(float speed);

  /// Retrieving the current speed
  float getSpeed();

  /// Retrieving the last mouse position
  void getMousePosition(int& x, int& y);

  /// Main function which is called to apply a camera motion.
  /// It is preferable to
  void motion(int x, int y, int action = 0);

  /// To call when the mouse wheel change
  void wheel(int value);

  /// Retrieve the screen width
  int getWidth() const;

  /// Retrieve the screen height
  int getHeight() const;

protected:
  Manipulator();

private:
  // Update the internal matrix.
  void update();
  // Do panning: movement parallels to the screen
  void pan(float dx, float dy);
  // Do orbiting: rotation around the center of interest. If invert, the interest orbit around the
  // camera position
  void orbit(float dx, float dy, bool invert = false);
  // Do dolly: movement toward the interest.
  void dolly(float dx, float dy);
  // Trackball: movement like the object is inside a ball
  void trackball(int x, int y);
  // Used by the trackball
  double projectOntoTBSphere(const glm::vec2& p);

protected:
  // Camera position
  glm::vec3 m_pos = glm::vec3(10, 10, 10);
  glm::vec3 m_int = glm::vec3(0, 0, 0);
  glm::vec3 m_up = glm::vec3(0, 1, 0);
  float m_roll = 0; // Rotation around the Z axis in RAD
  glm::mat4 m_matrix = glm::mat4(1);

  // Screen
  int m_width = 1;
  int m_height = 1;

  // Other
  float m_speed = 30;
  glm::vec2 m_mouse = glm::vec2(0, 0);

  bool m_button = false; // Button pressed
  bool m_moving = false; // Mouse is moving
  float m_tbsize = 0.8f; // Trackball size;

  Modes m_mode = Examine;
};

// Global Manipulator
#define CameraManip Manipulator::Singleton()

//-----------------------------------------------------------------------------
// MATH functions
//
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
areEqual(T x, T y,
         int ulp = 2 // Units in the Last Place
)
{
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  // See: http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
  return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         // unless the result is subnormal
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type areDifferent(T x, T y,
                                                                                      int ulp = 2)
{
  return !areEqual(x, y, ulp);
}

template <typename T> bool isZero(const T& _a)
{
  return fabs(_a) < std::numeric_limits<T>::epsilon();
}
template <typename T> bool isOne(const T& _a)
{
  return areEqual(_a, (T)1);
}

inline float sign(float s)
{
  return (s < 0.f) ? -1.f : 1.f;
}
inline double sign(double s)
{
  return (s < 0.0) ? -1.0 : 1.0;
}

} // namespace nv_helpers_dx12
