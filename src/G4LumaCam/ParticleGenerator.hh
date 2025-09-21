#ifndef PARTICLE_GENERATOR_HH
#define PARTICLE_GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4GeneralParticleSource.hh"

class ParticleGenerator : public G4VUserPrimaryGeneratorAction {
public:
    ParticleGenerator();
    ~ParticleGenerator() override;

    void GeneratePrimaries(G4Event* anEvent) override;
    G4double getParticleEnergy() const { return lastEnergy; }

private:
    G4GeneralParticleSource* source;
    G4double lastEnergy;
};

#endif
