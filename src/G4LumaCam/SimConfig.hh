#ifndef SIM_CONFIG_HH
#define SIM_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "G4String.hh"
#include <random>

namespace Sim {
    extern G4String outputFileName;
    extern G4int batchSize;
    extern std::default_random_engine randomEngine;
    
    constexpr G4double WORLD_SIZE = 50.0 * m;
    constexpr G4double SCINT_THICKNESS = 1.0 * cm; // half thickness
    constexpr G4double SAMPLE_THICKNESS = 3.75 * cm; // half thickness
    constexpr G4double SCINT_SIZE = 6.0 * cm; // half size
    constexpr G4double COATING_THICKNESS = 0.01 * cm; // half thickness
}

#endif