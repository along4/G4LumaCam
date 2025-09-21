#ifndef SIM_CONFIG_HH
#define SIM_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "G4String.hh"
#include <random>

// This header should ONLY contain declarations, not definitions

namespace Sim {
    // Declare extern variables
    extern G4String outputFileName;
    extern G4int batchSize;
    extern std::default_random_engine randomEngine;
    extern G4double WORLD_SIZE;
    extern G4double SCINT_THICKNESS; // half thickness
    extern G4double SAMPLE_THICKNESS; // half thickness
    extern G4double SCINT_SIZE; // half size
    extern G4double COATING_THICKNESS; // half thickness

    // New time window variables
    extern G4double TMIN;
    extern G4double TMAX;

    // Function declarations
    void SetScintThickness(G4double thickness);
    void SetSampleThickness(G4double thickness);
}

#endif // SIM_CONFIG_HH
