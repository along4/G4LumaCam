#include "EventProcessor.hh"
#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include <filesystem>

EventProcessor::EventProcessor(const G4String& name, ParticleGenerator* gen) 
    : G4VSensitiveDetector(name), neutronCount(-1), batchCount(0), eventCount(0), particleGen(gen), neutronRecorded(false) {
    resetData();
}

EventProcessor::~EventProcessor() {
    if (dataFile.is_open()) dataFile.close();
    if (triggerFile.is_open()) triggerFile.close();
}

void EventProcessor::Initialize(G4HCofThisEvent*) {
    resetData();
}

void EventProcessor::resetData() {
    photons.clear();
    tracks.clear();
    neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
    neutronEnergy = 0.;
    protonEnergy = 0.;
    lensPos[0] = lensPos[1] = 0.;
    neutronRecorded = false;
    neutronTriggerTimes.clear(); // Clear trigger times
}

G4bool EventProcessor::ProcessHits(G4Step* step, G4TouchableHistory*) {
    G4Track* track = step->GetTrack();
    G4String volName = track->GetVolume()->GetName();
    G4String particleName = track->GetDefinition()->GetParticleName();
    G4StepPoint* preStep = step->GetPreStepPoint();
    G4StepPoint* postStep = step->GetPostStepPoint();
    G4ThreeVector prePos = preStep->GetPosition();
    G4ThreeVector postPos = postStep->GetPosition();
    G4ThreeVector preDir = preStep->GetMomentumDirection();
    G4int tid = track->GetTrackID();
    G4int parentID = track->GetParentID();

    // Record primary particle's first interaction in ScintPhys
    if (volName == "ScintPhys" && parentID == 0 && !neutronRecorded) {
        G4String processName = postStep->GetProcessDefinedStep() ? 
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName != "Transportation") {
            neutronPos[0] = postPos.x();
            neutronPos[1] = postPos.y();
            neutronPos[2] = postPos.z();
            neutronEnergy = particleGen ? particleGen->getParticleEnergy() : track->GetKineticEnergy() / MeV;
            neutronCount++;
            neutronRecorded = true;
            neutronTriggerTimes.push_back(track->GetGlobalTime() / ps); // Store trigger time in ps
        }
    }

    if (volName == "ScintPhys" && particleName != "opticalphoton") {
        if (tracks.find(tid) == tracks.end()) {
            G4double energy = track->GetKineticEnergy() / MeV;
            if (parentID != 0 && energy <= 0) {
                energy = neutronEnergy;
            }
            tracks[tid] = {particleName, prePos.x(), prePos.y(), prePos.z(), energy, false};
        }

        G4String processName = postStep->GetProcessDefinedStep() ? 
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName == "Scintillation" || processName == "Cerenkov") {
            tracks[tid].x = prePos.x();
            tracks[tid].y = prePos.y();
            tracks[tid].z = prePos.z();
            tracks[tid].isLightProducer = true;
            G4double currentEnergy = track->GetKineticEnergy() / MeV;
            if (currentEnergy <= 0 && tracks[tid].energy <= 0) {
                tracks[tid].energy = neutronEnergy;
            }
        }
    }

    if (volName == "MonitorPhys" && particleName == "opticalphoton") {
        lensPos[0] = postStep->GetPosition().x() / mm + 500. * preStep->GetMomentumDirection().x();
        lensPos[1] = postStep->GetPosition().y() / mm + 500. * preStep->GetMomentumDirection().y();

        if (lensPos[0] > -27.5 && lensPos[0] < 27.5 && lensPos[1] > -27.5 && lensPos[1] < 27.5) {
            if (tracks.find(parentID) == tracks.end()) {
                tracks[parentID] = {"unknown", neutronPos[0], neutronPos[1], neutronPos[2], neutronEnergy, true};
            }
            
            if (tracks[parentID].energy <= 0) {
                tracks[parentID].energy = neutronEnergy;
            }
            
            PhotonRecord rec;
            rec.id = track->GetTrackID();
            rec.parentId = parentID;
            rec.neutronId = neutronCount;
            rec.x = prePos.x() / mm;
            rec.y = prePos.y() / mm;
            rec.z = 0.; 
            rec.dx = preDir.x();
            rec.dy = preDir.y();
            rec.dz = preDir.z();
            rec.timeOfArrival = track->GetGlobalTime() / ns;
            rec.wavelength = 1240. / (track->GetTotalEnergy() / eV);
            rec.parentType = tracks[parentID].type;
            rec.px = tracks[parentID].x / mm;
            rec.py = tracks[parentID].y / mm;
            rec.pz = tracks[parentID].z / mm;
            rec.parentEnergy = tracks[parentID].energy;
            rec.nx = neutronPos[0] / mm;
            rec.ny = neutronPos[1] / mm;
            rec.nz = neutronPos[2] / mm;
            rec.neutronEnergy = neutronEnergy;
            
            photons.push_back(rec);
        }
    }
    return true;
}

void EventProcessor::EndOfEvent(G4HCofThisEvent*) {
    if (eventCount == 0 && batchCount == 0) {
        openOutputFile();
        openTriggerFile();
    }
    if (!photons.empty()) writeData();
    if (!neutronTriggerTimes.empty()) writeTriggerData();
    
    if (Sim::batchSize > 0) {
        eventCount++;
        if (eventCount >= Sim::batchSize) {
            batchCount++;
            eventCount = 0;
            openOutputFile();
            openTriggerFile();
        }
    }
    resetData();
}

void EventProcessor::openOutputFile() {
    if (dataFile.is_open()) dataFile.close();

    G4String fileName = Sim::outputFileName;
    size_t csvPos = fileName.find(".csv");
    if (csvPos != G4String::npos) {
        fileName = fileName.substr(0, csvPos);
    }

    if (Sim::batchSize > 0) {
        fileName += "_" + std::to_string(batchCount) + ".csv";
    } else {
        fileName += ".csv";
    }
    
    dataFile.open(fileName);
    dataFile << "id,parent_id,neutron_id,x,y,z,dx,dy,dz,toa,wavelength,"
             << "parentName,px,py,pz,parentEnergy,nx,ny,nz,neutronEnergy\n";
}

void EventProcessor::openTriggerFile() {
    if (triggerFile.is_open()) triggerFile.close();

    // Create TriggerTimes directory
    std::filesystem::create_directory("TriggerTimes");

    G4String fileName = "TriggerTimes/trigger_data";
    if (Sim::batchSize > 0) {
        fileName += "_" + std::to_string(batchCount) + ".csv";
    } else {
        fileName += ".csv";
    }

    triggerFile.open(fileName);
    triggerFile << "neutron_id,trigger_time_ps\n";
}

void EventProcessor::writeTriggerData() {
    for (size_t i = 0; i < neutronTriggerTimes.size(); ++i) {
        triggerFile << (neutronCount - neutronTriggerTimes.size() + i + 1) << ","
                    << neutronTriggerTimes[i] << "\n";
    }
    triggerFile.flush();
}

void EventProcessor::writeData() {
    for (const auto& p : photons) {
        dataFile << p.id << "," 
                 << p.parentId << "," 
                 << p.neutronId << ","
                 << p.x << "," 
                 << p.y << "," 
                 << p.z << ","
                 << p.dx << "," 
                 << p.dy << "," 
                 << p.dz << ","
                 << p.timeOfArrival << "," 
                 << p.wavelength << "," 
                 << p.parentType << ","
                 << p.px << "," 
                 << p.py << "," 
                 << p.pz << "," 
                 << p.parentEnergy << ","
                 << p.nx << "," 
                 << p.ny << "," 
                 << p.nz << "," 
                 << p.neutronEnergy << "\n";
    }
    dataFile.flush();
}