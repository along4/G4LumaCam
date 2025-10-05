#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Neutron.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

ParticleGenerator::ParticleGenerator()
    : source(new G4GeneralParticleSource()), lastEnergy(0.), currentPulseIndex(0) {
    source->SetParticleDefinition(G4Neutron::NeutronDefinition());
}

ParticleGenerator::~ParticleGenerator() {
    delete source;
}

void ParticleGenerator::SetTotalNeutrons(G4int totalNeutrons) {
    Sim::ComputePulseStructure(totalNeutrons);
    currentPulseIndex = 0; // Reset pulse index
}

void ParticleGenerator::GeneratePrimaries(G4Event* anEvent) {
    source->GeneratePrimaryVertex(anEvent);

    // Check if pulse structure is defined
    if (!Sim::pulseTimes.empty() && Sim::FREQ > 0 && Sim::FLUX > 0) {
        // Assign neutron to the current pulse
        if (currentPulseIndex < Sim::pulseTimes.size()) {
            G4double t0 = Sim::pulseTimes[currentPulseIndex];
            anEvent->GetPrimaryVertex()->SetT0(t0 * ps); // Set time in ps
            // Move to next pulse when all neutrons for current pulse are assigned
            static G4int neutronsInCurrentPulse = 0;
            neutronsInCurrentPulse++;
            if (neutronsInCurrentPulse >= Sim::neutronsPerPulse[currentPulseIndex]) {
                currentPulseIndex++;
                neutronsInCurrentPulse = 0;
            }
        } else {
            // Fallback if pulse index exceeds available pulses
            anEvent->GetPrimaryVertex()->SetT0(0.0 * ns);
        }
    } else if (Sim::TMAX > Sim::TMIN) {
        G4double t0 = Sim::TMIN + (Sim::TMAX - Sim::TMIN) * G4UniformRand();
        anEvent->GetPrimaryVertex()->SetT0(t0);
    } else if (Sim::TMIN > 0.0) {
        anEvent->GetPrimaryVertex()->SetT0(Sim::TMIN);
    } else {
        anEvent->GetPrimaryVertex()->SetT0(0.0 * ns);
    }

    lastEnergy = source->GetParticleEnergy() / MeV;
}