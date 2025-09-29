#pragma once
// -- This file contains central config options for mainly debugging purposes --

// --- main program ---
// #define SPDLOG_LOG_TYPE 1 // 0/undefined => disable; 1 => console; 2 => logfile
// #define LOG_CANDIDATES_DETAILED
// #define MAT_DMP
// #define ASSERT_SOLVERS

// --- bfs.cpp ---
// #define DBG_BFS 1
#define BFS_TRUST_LB

// --- matrix.cpp ---
// #define DBG_MAT_DETAIL
// #define DBG_MAT
// #define DBG_MAT_BRIEF
// #define DBG_RHS
// #define DBG_SLACK_N
// #define MAT_DONT_TRUST_BFS
// #define RHS_TRUST_A_REMOVAL // Trust Lemma 16 n' definition. Possibly only meant for Santa Claus, not Makespan Minimization

// --- gurobi_env.cpp ---
// #define DBG_GRB // gurobi enable solving output
// #define DGB_GRB_DET // gurobi improve determinism/reproducability

// --- feasibility.cpp, arrow_climb.cpp ---
// #define DBG_AC 1
// #define AC_USE_DIRECT_SOLVE

// --- bt_enumerator.cpp ---
// #define DBG_BT       1  // high-level enter/exit + shapes
// #define DBG_CAPS     1  // show repCap and caps
// #define DBG_COLS     1  // show usable column filtering summary
// #define DBG_LEVELS   1  // per-level sizes and a few samples
// #define DBG_S0       1  // s==0 path decisions

// --- pooled_solver.cpp ---
// #define DBG_POOLED_SOLVER