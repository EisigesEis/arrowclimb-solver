#include "gurobi_env.h"
#include "Config.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

static std::once_flag env_once;
static std::unique_ptr<GRBEnv> env_ptr;

GRBEnv &grb_global_env() {
  std::call_once(env_once, [] {
    try {
      env_ptr = std::make_unique<GRBEnv>(true);

#ifdef DGB_GRB_DET
      env_ptr->set(GRB_IntParam_Threads, 1);
      env_ptr->set(GRB_IntParam_Seed, 0);
      env_ptr->set(GRB_DoubleParam_IntFeasTol, 1e-9);
      env_ptr->set(GRB_DoubleParam_MIPGap, 0.0);
#endif

#ifdef DBG_GRB
      env_ptr->set(GRB_IntParam_OutputFlag, 1);
#else
      env_ptr->set(GRB_IntParam_OutputFlag, 0);
#endif

      env_ptr->start();
    } catch (const GRBException &e) {
      env_ptr.reset();
      throw std::runtime_error(
          std::string("Gurobi init failed (code ") + std::to_string(e.getErrorCode()) + "): " + e.getMessage());
    } catch (const std::exception &e) {
      env_ptr.reset();
      throw std::runtime_error(std::string("Gurobi init failed: ") + e.what());
    }
  });
  return *env_ptr;
}