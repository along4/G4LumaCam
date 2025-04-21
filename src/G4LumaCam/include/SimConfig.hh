#ifndef SIMCONFIG_HH
#define SIMCONFIG_HH

#include <string>
#include <random>
#include "G4SystemOfUnits.hh" // Include Geant4 units

class Sim {
public:
    static std::string outputFileName;
    static int batchSize;
    static std::default_random_engine randomEngine;

    // Add the missing static members with proper units
    static constexpr double SCINT_SIZE = 5.0 * CLHEP::cm;          // Example value
    static constexpr double SAMPLE_THICKNESS = 1.0 * CLHEP::cm;    // Example value
    static constexpr double WORLD_SIZE = 200.0 * CLHEP::cm;         // Example value
    static constexpr double SCINT_THICKNESS = 0.5 * CLHEP::cm;     // Example value
    static constexpr double COATING_THICKNESS = 0.1 * CLHEP::cm;   // Example value
};

#endif