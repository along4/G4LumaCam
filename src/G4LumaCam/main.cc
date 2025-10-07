#include "SimConfig.hh"
#include "GeometryConstructor.hh"
#include "ParticleGenerator.hh"
#include "SimulationManager.hh"
#include "EventProcessor.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "QGSP_BERT_HP.hh"
#include "G4OpticalPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"

int main(int argc, char** argv) {
    Sim::batchSize = 10000;
    
    G4RunManager* runMgr = new G4RunManager();
    
    G4VModularPhysicsList* phys = new QGSP_BERT_HP();
    G4OpticalPhysics* optPhys = new G4OpticalPhysics();
    optPhys->Configure(kCerenkov, true);
    optPhys->Configure(kScintillation, true);
    phys->RegisterPhysics(optPhys);
    phys->RegisterPhysics(new G4RadioactiveDecayPhysics());
    runMgr->SetUserInitialization(phys);
    
    ParticleGenerator* gen = new ParticleGenerator();
    GeometryConstructor* geo = new GeometryConstructor(gen);
    runMgr->SetUserInitialization(geo);
    runMgr->SetUserAction(gen);
    
    SimulationManager* simMgr = new SimulationManager();
    // simMgr->SetParticleGenerator(gen);  // IMPORTANT: Connect the generator
    runMgr->SetUserAction(simMgr);
    runMgr->SetUserAction(new SimulationManager::EventHandler(simMgr));
    
    runMgr->Initialize();
    
    G4VisManager* visMgr = new G4VisExecutive();
    visMgr->Initialize();
    
    G4UImanager* uiMgr = G4UImanager::GetUIpointer();
    
    // Set total neutrons BEFORE running (important for pulse structure)
    // This can be overridden by macro commands or user input
    G4int defaultNeutrons = 1000;  // Default value
    simMgr->SetTotalNeutrons(defaultNeutrons);
    G4cout << "Default total neutrons set to: " << defaultNeutrons << G4endl;
    G4cout << "(This will be used when /run/beamOn is called)" << G4endl;
    
    if (argc > 1) {
        // Macro mode - execute commands from file
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        uiMgr->ApplyCommand(command + fileName);
    } else {
        // Interactive mode
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);
        
        // Visualization setup
        uiMgr->ApplyCommand("/vis/open OGL");
        uiMgr->ApplyCommand("/vis/drawVolume");
        uiMgr->ApplyCommand("/vis/scene/add/trajectories");
        uiMgr->ApplyCommand("/vis/viewer/set/background white");
        uiMgr->ApplyCommand("/vis/viewer/set/lineWidth 4");
        
        // Particle source setup
        uiMgr->ApplyCommand("/gps/direction 0 0 1");
        uiMgr->ApplyCommand("/gps/position 0 0 -1085 cm");
        uiMgr->ApplyCommand("/gps/energy 10 MeV");
        uiMgr->ApplyCommand("/gps/particle neutron");
        
        // Material and beam setup
        uiMgr->ApplyCommand("/lumacam/sampleMaterial G4_Galactic");
        uiMgr->ApplyCommand("/lumacam/scintMaterial EJ200");
        uiMgr->ApplyCommand("/lumacam/flux 1e4");
        uiMgr->ApplyCommand("/lumacam/freq 200000");
        
        // IMPORTANT: Set up pulse structure AFTER flux and freq are set
        // This should be done before running, but since we're in interactive mode,
        // we'll create a custom command or do it before each run
        G4cout << "\n=== Initial Pulse Structure Setup (Interactive Mode) ===" << G4endl;
        G4cout << "Note: Call /run/beamOn N to start simulation" << G4endl;
        G4cout << "Pulse structure will be computed when you run." << G4endl;
        G4cout << "========================================================\n" << G4endl;
        
        // Trajectory filtering
        uiMgr->ApplyCommand("/vis/filtering/trajectories/particleFilter-0/add proton");
        uiMgr->ApplyCommand("/vis/filtering/trajectories/particleFilter-0/add opticalphoton");
        uiMgr->ApplyCommand("/vis/filtering/trajectories/particleFilter-0/add neutron");
        uiMgr->ApplyCommand("/vis/filtering/trajectories/particleFilter-0/add e-");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/create/drawByParticleID");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/setLineWidth 2");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/setRGBA proton 1.0 0.0 0.0 0.6");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/setRGBA opticalphoton 0.8 0.2 1.0 0.3");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/setRGBA neutron 0.0 1.0 1.0 0.6");
        uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/setRGBA e- 0.0 1.0 0.0 0.6");
        
        ui->SessionStart();
        delete ui;
        ui = nullptr;
    }
    
    if (visMgr) {
        delete visMgr;
        visMgr = nullptr;
    }
    if (runMgr) {
        delete runMgr;
        runMgr = nullptr;
    }
    
    return 0;
}