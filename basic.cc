#include <cstdio>
#include <cstring> 
#include <cassert>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <eigen3/Eigen/Dense>
// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// main function
int main(int argc, const char** argv) {
  // check command-line arguments
  //if (argc!=2) {
  //  std::printf(" USAGE:  basic modelfile\n");
  //  return 0;
  //}

  // load and compile model
  char error[1000] = "Could not load binary model";
  //if (std::strlen(argv[1])>4 && !std::strcmp(argv[1]+std::strlen(argv[1])-4, ".mjb")) {
  //  m = mj_loadModel(argv[1], 0);
  //} else {
  //  m = mj_loadXML(argv[1], 0, error, 1000);
  //}

  m = mj_loadXML("../model/franka_fr3/scene.xml", 0, error, 1000);
  if (!m) {
    mju_error("Load model error: %s", error);
  }

  // make data
  d = mj_makeData(m);
  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  //define visualization flags
  opt.flags[mjVIS_JOINT] = 1;
  opt.flags[mjVIS_TRANSPARENT] = 1;


  // create window, make OpenGL context current, request v-sync
  GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);


// Assert that the size of qpos is 7 by checking the model's nq (number of generalized coordinates)
    // Output nq (size of qpos)
  std::cout << "Size of qpos (nq): " << m->nq << std::endl;
  std::cout << "Size of qpos (nv): " << m->nv << std::endl;
  //assert(m->nq == 7 && "qpos size is not 7!");

  Eigen::Vector3d x_d(0.5,0.0, 0.6);
  double tol = 0.01;
  double damp = 0.0;
  Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(d->qpos, m->nv);
  Eigen::Vector3d e;
  double alpha = 0.001;     //learning rate of the solver
  int ee_id = mj_name2id(m, mjOBJ_BODY, "fr3_link7");
  mjtNum jacp[3 * m->nv];
  size_t it = 0;
  while (++it < 6000) {  
    mj_kinematics(m, d);    // run forward kinematics  
    mj_comPos(m, d);

    Eigen::Vector3d x(d->xpos[3 * ee_id], d->xpos[3 * ee_id + 1], d->xpos[3 * ee_id + 2]); // compute x = f(q)
    e = x_d-x;
    std::cout << "x: " << x << std::endl;

    if (e.norm() < tol) {
        break;
    }
    //mjtNum goal[3];
    //goal[0] = x[0];
    //goal[1] = x[1];
    //goal[2] = x[2];    
    //mj_jac(m, d, jacp, NULL, goal, ee_id);
    mj_jacBody(m, d, jacp, NULL, ee_id);
    Eigen::Map<const Eigen::Matrix<mjtNum, 3, Eigen::Dynamic>> J(jacp, 3, m->nv);
    std::cout << "Jacobian Matrix J (3x9):\n" << J << std::endl;

    Eigen::VectorXd delta_q = alpha * (J.transpose() * ((J * J.transpose()+ damp*Eigen::Matrix3d::Identity()).inverse())) * e; 
    q += delta_q;

    for (size_t i = 0; i < m->nv; i++) {
        q[i] = std::max(m->jnt_range[2*i], std::min(q[i], m->jnt_range[2*i+1]));      //check for joint limits
        d->qpos[i] = q[i];  // update joint angles 
        //std::cout << "Q: "<< q[i] << std::endl;
    }


    // Render the updated state
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);  // swap OpenGL buffers
    glfwPollEvents();         // process GUI events
  }

    std::cout << "Final error (norm): " << e.norm() << std::endl;

// Only render the current state of the robot after optimization
while (!glfwWindowShouldClose(window)) {
  // get framebuffer viewport
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

  // update scene and render the final state
  mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
  mjr_render(viewport, &scn, &con);

  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(window);

  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();
}

// free visualization storage
mjv_freeScene(&scn);
mjr_freeContext(&con);

// free MuJoCo model and data
mj_deleteData(d);
mj_deleteModel(m);

// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
glfwTerminate();
#endif

  return 1;
}