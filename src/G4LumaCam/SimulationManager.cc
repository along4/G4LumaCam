#include "SimulationManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

SimulationManager::SimulationManager() : processor(new EventProcessor("Tracker")), eventCounter(0) {}

void SimulationManager::BeginOfRunAction(const G4Run*) {
    eventCounter = 0;
}

void SimulationManager::EndOfRunAction(const G4Run*) {}

SimulationManager::EventHandler::EventHandler(SimulationManager* mgr) : manager(mgr) {}

void SimulationManager::EventHandler::BeginOfEventAction(const G4Event*) {}

void SimulationManager::EventHandler::EndOfEventAction(const G4Event* evt) {
    manager->eventCounter++;
    // if (manager->eventCounter % 100 == 0) {
    //     G4cout << "Processing event: " << manager->eventCounter << G4endl;
    // }
}