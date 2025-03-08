#include "ParticleGenerator.hh"
#include "G4Neutron.hh"
#include "G4SystemOfUnits.hh"

ParticleGenerator::ParticleGenerator() : source(new G4GeneralParticleSource()), lastEnergy(0.) {
    source->SetParticleDefinition(G4Neutron::NeutronDefinition());
}

ParticleGenerator::~ParticleGenerator() {
    delete source;
}

void ParticleGenerator::GeneratePrimaries(G4Event* anEvent) {
    source->GeneratePrimaryVertex(anEvent);
    lastEnergy = source->GetParticleEnergy() / MeV;
}