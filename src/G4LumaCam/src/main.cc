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

G4String Sim::outputFileName = "sim_data.csv";
G4int Sim::batchSize = 10000;
std::default_random_engine Sim::randomEngine(time(nullptr));

int main(int argc, char** argv) {
    G4RunManager* runMgr = new G4RunManager();

    // Step 1: Set up the physics list first
    G4VModularPhysicsList* phys = new QGSP_BERT_HP();
    G4OpticalPhysics* optPhys = new G4OpticalPhysics();
    optPhys->Configure(kCerenkov, true);
    optPhys->Configure(kScintillation, true);
    phys->RegisterPhysics(optPhys);
    phys->RegisterPhysics(new G4RadioactiveDecayPhysics());
    runMgr->SetUserInitialization(phys);

    // Step 2: Now instantiate user actions
    ParticleGenerator* gen = new ParticleGenerator();
    GeometryConstructor* geo = new GeometryConstructor(gen); // Pass gen to GeometryConstructor
    runMgr->SetUserInitialization(geo);

    runMgr->SetUserAction(gen);

    SimulationManager* simMgr = new SimulationManager();
    runMgr->SetUserAction(simMgr);
    runMgr->SetUserAction(new SimulationManager::EventHandler(simMgr));

    runMgr->Initialize();

    G4VisManager* visMgr = new G4VisExecutive();
    visMgr->Initialize();

    G4UImanager* uiMgr = G4UImanager::GetUIpointer();
    if (argc > 1) {
        uiMgr->ApplyCommand("/control/execute " + G4String(argv[1]));
    } else {
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);
        uiMgr->ApplyCommand("/vis/open OGL");
        uiMgr->ApplyCommand("/vis/drawVolume");
        uiMgr->ApplyCommand("/vis/scene/add/trajectories");
        uiMgr->ApplyCommand("/vis/viewer/set/background white");
        uiMgr->ApplyCommand("/vis/viewer/set/lineWidth 4");
        uiMgr->ApplyCommand("/gps/direction 0 0 1");
        uiMgr->ApplyCommand("/gps/position 0 0 -1059 cm");
        uiMgr->ApplyCommand("/gps/energy 10 MeV");
        uiMgr->ApplyCommand("/gps/particle neutron");
        uiMgr->ApplyCommand("/lumacam/sampleMaterial G4_Galactic");
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
    }

    delete visMgr;
    delete runMgr;
    return 0;
}