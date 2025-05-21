#include "SimConfig.hh"
#include <ctime>
#include "G4ios.hh"  // For G4cout, G4cerr, G4endl

// This file contains all the definitions for what was declared in the header

namespace Sim {
    // Define and initialize variables here - these are the definitive instances
    G4String outputFileName = "sim_data.csv";
    G4int batchSize = 0;  // This will be set to 10000 in main.cc
    std::default_random_engine randomEngine(time(nullptr));
    G4double WORLD_SIZE = 50.0 * m;
    G4double SCINT_THICKNESS = 1.0 * cm; // half thickness
    G4double SAMPLE_THICKNESS = 3.75 * cm; // half thickness
    G4double SCINT_SIZE = 6.0 * cm; // half size
    G4double COATING_THICKNESS = 0.01 * cm; // half thickness

    // Function implementations
    void SetScintThickness(G4double thickness) {
        if (thickness > 0) {
            SCINT_THICKNESS = thickness;
            G4cout << "Scintillator thickness set to: " << thickness / cm << " cm" << G4endl;
        } else {
            G4cerr << "ERROR: Scintillator thickness must be positive!" << G4endl;
        }
    }

    void SetSampleThickness(G4double thickness) {
        if (thickness > 0) {
            SAMPLE_THICKNESS = thickness;
            G4cout << "Sample thickness set to: " << thickness / cm << " cm" << G4endl;
        } else {
            G4cerr << "ERROR: Sample thickness must be positive!" << G4endl;
        }
    }
}