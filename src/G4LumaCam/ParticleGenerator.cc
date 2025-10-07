#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Neutron.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

ParticleGenerator::ParticleGenerator()
    : source(new G4GeneralParticleSource()), lastEnergy(0.), 
      currentPulseIndex(0), neutronsInCurrentPulse(0) {
    source->SetParticleDefinition(G4Neutron::NeutronDefinition());
}

ParticleGenerator::~ParticleGenerator() {
    delete source;
}

void ParticleGenerator::SetTotalNeutrons(G4int totalNeutrons) {
    Sim::ComputePulseStructure(totalNeutrons);
    currentPulseIndex = 0;
    neutronsInCurrentPulse = 0;
    
    G4cout << "=== Pulse Structure Debug ===" << G4endl;
    G4cout << "Total pulses: " << Sim::pulseTimes.size() << G4endl;
    G4cout << "Total neutrons assigned: ";
    G4int totalAssigned = 0;
    for (size_t i = 0; i < Sim::neutronsPerPulse.size(); ++i) {
        totalAssigned += Sim::neutronsPerPulse[i];
    }
    G4cout << totalAssigned << G4endl;
    
    for (size_t i = 0; i < std::min(size_t(10), Sim::pulseTimes.size()); ++i) {
        G4cout << "Pulse " << i << ": t=" << Sim::pulseTimes[i] 
               << " ps, n=" << Sim::neutronsPerPulse[i] << G4endl;
    }
    if (Sim::pulseTimes.size() > 10) {
        G4cout << "... (" << (Sim::pulseTimes.size() - 10) << " more pulses)" << G4endl;
    }
    G4cout << "=============================" << G4endl;
}

void ParticleGenerator::GeneratePrimaries(G4Event* anEvent) {
    source->GeneratePrimaryVertex(anEvent);
    
    // Check if pulse structure is defined
    if (!Sim::pulseTimes.empty() && Sim::FREQ > 0 && Sim::FLUX > 0) {
        if (currentPulseIndex < Sim::pulseTimes.size()) {
            G4double t0 = Sim::pulseTimes[currentPulseIndex];
            anEvent->GetPrimaryVertex()->SetT0(t0 * ns);
            
            // Debug output (reduce frequency to avoid spam)
            if (neutronsInCurrentPulse == 0) {
                G4cout << ">>> Starting pulse " << currentPulseIndex 
                       << " at t=" << t0 << " ns with " 
                       << Sim::neutronsPerPulse[currentPulseIndex] 
                       << " neutrons" << G4endl;
            }
            
            neutronsInCurrentPulse++;
            
            // Debug: show progress through pulse
            if (neutronsInCurrentPulse % 100 == 0 || 
                neutronsInCurrentPulse == Sim::neutronsPerPulse[currentPulseIndex]) {
                G4cout << "    Pulse " << currentPulseIndex 
                       << " progress: " << neutronsInCurrentPulse 
                       << "/" << Sim::neutronsPerPulse[currentPulseIndex] << G4endl;
            }
            
            // Move to next pulse when current pulse is complete
            if (neutronsInCurrentPulse >= Sim::neutronsPerPulse[currentPulseIndex]) {
                currentPulseIndex++;
                neutronsInCurrentPulse = 0;
            }
        } else {
            // No more pulses defined
            anEvent->GetPrimaryVertex()->SetT0(0.0 * ns);
            G4cerr << "WARNING: Exceeded available pulses!" << G4endl;
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