#include "SimConfig.hh"
#include <ctime>
#include "G4ios.hh"
#include <cmath>
#include <filesystem>
#include "Randomize.hh"

namespace Sim {
    G4String outputFileName = "sim_data.csv";
    G4int batchSize = 0;
    std::default_random_engine randomEngine(time(nullptr));
    G4double WORLD_SIZE = 50.0 * m;
    G4double SCINT_THICKNESS = 1.0 * cm;
    G4double SAMPLE_THICKNESS = 3.75 * cm;
    G4double SCINT_SIZE = 6.0 * cm;
    G4double COATING_THICKNESS = 0.01 * cm;
    G4double TMIN = 0.0 * ns;
    G4double TMAX = 0.0 * ns;
    G4double FLUX = 0.0; // Default: no pulsed structure
    G4double FREQ = 0.0; // Default: no pulsed structure
    std::vector<G4double> pulseTimes; // Trigger times in ps
    std::vector<G4int> neutronsPerPulse; // Neutrons per pulse

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

    void ComputePulseStructure(G4int totalNeutrons) {
        pulseTimes.clear();
        neutronsPerPulse.clear();

        if (FLUX <= 0 || FREQ <= 0 || totalNeutrons <= 0) {
            G4cout << "Pulse structure not computed: FLUX=" << FLUX << ", FREQ=" << FREQ
                   << ", totalNeutrons=" << totalNeutrons << G4endl;
            return;
        }

        // Calculate FOV area (cm²)
        G4double fovArea = (2 * SCINT_SIZE / cm) * (2 * SCINT_SIZE / cm);
        G4double neutronsPerSecond = FLUX * fovArea; // Total neutrons/s
        G4double pulsePeriod = 1.0 / FREQ; // Seconds per pulse
        G4double avgNeutronsPerPulse = neutronsPerSecond / FREQ; // Neutrons per pulse
        G4int numPulses = std::ceil(totalNeutrons / avgNeutronsPerPulse);

        G4cout << "Computing pulse structure: FOV=" << fovArea << " cm², "
               << "Neutrons/s=" << neutronsPerSecond << ", "
               << "Pulse period=" << pulsePeriod / us << " us, "
               << "Avg neutrons/pulse=" << avgNeutronsPerPulse << ", "
               << "Total pulses=" << numPulses << G4endl;

        G4int remainingNeutrons = totalNeutrons;
        for (G4int i = 0; i < numPulses && remainingNeutrons > 0; ++i) {
            // Randomize neutrons per pulse to maintain average flux
            G4double rand = G4UniformRand();
            G4int neutronsThisPulse = std::floor(avgNeutronsPerPulse + rand);
            if (neutronsThisPulse > remainingNeutrons) {
                neutronsThisPulse = remainingNeutrons;
            }
            if (neutronsThisPulse < 0) neutronsThisPulse = 0;

            neutronsPerPulse.push_back(neutronsThisPulse);
            pulseTimes.push_back(i * pulsePeriod * s / ps); // Convert to ps
            remainingNeutrons -= neutronsThisPulse;
        }

        // Ensure total neutrons match by adjusting last pulse if needed
        if (remainingNeutrons < 0 && !neutronsPerPulse.empty()) {
            neutronsPerPulse.back() += remainingNeutrons;
        }

        G4cout << "Pulse structure computed: " << pulseTimes.size() << " pulses, "
               << totalNeutrons - remainingNeutrons << " neutrons assigned" << G4endl;
    }
}