#include "SimConfig.hh"
#include <ctime>

// Define static members
std::string Sim::outputFileName = "sim_data.csv";
int Sim::batchSize = 10000;
std::default_random_engine Sim::randomEngine(time(nullptr));