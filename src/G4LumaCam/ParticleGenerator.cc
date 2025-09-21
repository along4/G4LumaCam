#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Neutron.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

ParticleGenerator::ParticleGenerator()
    : source(new G4GeneralParticleSource()), lastEnergy(0.) {
    source->SetParticleDefinition(G4Neutron::NeutronDefinition());
}

ParticleGenerator::~ParticleGenerator() {
    delete source;
}

void ParticleGenerator::GeneratePrimaries(G4Event* anEvent) {
    // Let GPS handle position, energy, direction
    source->GeneratePrimaryVertex(anEvent);

    // Uniform random emission time if requested
    if (Sim::TMAX > Sim::TMIN) {
        G4double t0 = Sim::TMIN + (Sim::TMAX - Sim::TMIN) * G4UniformRand();
        anEvent->GetPrimaryVertex()->SetT0(t0);
    } else if (Sim::TMIN > 0.0) {
        // Fixed time if tmin==tmax>0
        anEvent->GetPrimaryVertex()->SetT0(Sim::TMIN);
    } else {
        // Default = 0 ns
        anEvent->GetPrimaryVertex()->SetT0(0.0 * ns);
    }

    lastEnergy = source->GetParticleEnergy() / MeV;
}
