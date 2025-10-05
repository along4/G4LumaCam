#include "SimulationManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "ParticleGenerator.hh"
#include "G4UnitsTable.hh"

SimulationManager::SimulationManager() : processor(new EventProcessor("Tracker")), eventCounter(0), totalNeutrons(0) {}

void SimulationManager::BeginOfRunAction(const G4Run* run) {
    eventCounter = 0;
    // Set total neutrons in ParticleGenerator
    G4RunManager* runManager = G4RunManager::GetRunManager();
    ParticleGenerator* generator = dynamic_cast<ParticleGenerator*>(
        const_cast<G4VUserPrimaryGeneratorAction*>(runManager->GetUserPrimaryGeneratorAction()));
    if (generator && totalNeutrons > 0) {
        generator->SetTotalNeutrons(totalNeutrons);
    }
}

void SimulationManager::EndOfRunAction(const G4Run*) {}

void SimulationManager::SetTotalNeutrons(G4int totalNeutrons) {
    this->totalNeutrons = totalNeutrons;
}

SimulationManager::EventHandler::EventHandler(SimulationManager* mgr) : manager(mgr) {}

void SimulationManager::EventHandler::BeginOfEventAction(const G4Event*) {}

void SimulationManager::EventHandler::EndOfEventAction(const G4Event* evt) {
    manager->eventCounter++;
}