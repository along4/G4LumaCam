#include "EventProcessor.hh"
#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include <filesystem>
#include <cstdlib>
#include <set>

EventProcessor::EventProcessor(const G4String& name, ParticleGenerator* gen) 
    : G4VSensitiveDetector(name), neutronCount(-1), batchCount(0), eventCount(0), 
      particleGen(gen), neutronRecorded(false), currentEventTriggerTime(-1.0) {
    resetData();
}

EventProcessor::~EventProcessor() {
    if (dataFile.is_open()) dataFile.close();
    if (triggerFile.is_open()) triggerFile.close();
}

void EventProcessor::Initialize(G4HCofThisEvent*) {
    resetData();
}

void EventProcessor::ClearRecordedTriggerTimes() {
    recordedTriggerTimes.clear();
    triggerTimeToPulseId.clear(); // Clear the trigger time to pulse ID mapping
}

void EventProcessor::resetData() {
    photons.clear();
    tracks.clear();
    neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
    neutronEnergy = 0.;
    protonEnergy = 0.;
    lensPos[0] = lensPos[1] = 0.;
    neutronRecorded = false;
    currentEventTriggerTime = -1.0; // Reset trigger time for this event
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

    // Set trigger time for every event
    if (!neutronRecorded) { // Only set once per event
        const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent();
        if (event && event->GetNumberOfPrimaryVertex() > 0) {
            currentEventTriggerTime = event->GetPrimaryVertex(0)->GetT0() / ns;
        }
    }

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
            rec.pulseId = triggerTimeToPulseId[currentEventTriggerTime]; // Assign pulse ID
            photons.push_back(rec);
        }
    }
    return true;
}

void EventProcessor::EndOfEvent(G4HCofThisEvent*) {
    if (eventCount == 0 && batchCount == 0) {
        G4cout << "EventProcessor: Starting new run with batchSize=" << Sim::batchSize << G4endl;
        openOutputFile();
        openTriggerFile();
    }
    
    if (!photons.empty()) writeData();
    
    // Write trigger time only once per pulse
    if (currentEventTriggerTime >= 0) {
        if (recordedTriggerTimes.find(currentEventTriggerTime) == recordedTriggerTimes.end()) {
            static G4int pulseId = 0; // Static to persist across batches
            writeTriggerData(currentEventTriggerTime, pulseId);
            triggerTimeToPulseId[currentEventTriggerTime] = pulseId; // Map trigger time to pulse ID
            recordedTriggerTimes.insert(currentEventTriggerTime);
            pulseId++;
        }
    }
    
    if (Sim::batchSize > 0) {
        eventCount++;
        G4cout << "EventProcessor: eventCount=" << eventCount << ", batchSize=" << Sim::batchSize << G4endl;
        if (eventCount >= Sim::batchSize) {
            batchCount++;
            eventCount = 0;
            recordedTriggerTimes.clear(); // Clear for new batch
            G4cout << "EventProcessor: Starting new batch " << batchCount << G4endl;
            openOutputFile();
            openTriggerFile();
        }
    }
    resetData();
}

void EventProcessor::openOutputFile() {
    if (dataFile.is_open()) dataFile.close();

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path simPhotonsDir = currentPath / "SimPhotons";
    
    try {
        std::filesystem::create_directories(simPhotonsDir);
        G4cout << "Created/verified directory: " << simPhotonsDir << G4endl;
    } catch (const std::filesystem::filesystem_error& e) {
        G4cerr << "ERROR: Failed to create directory " << simPhotonsDir << ": " << e.what() << G4endl;
        G4Exception("EventProcessor::openOutputFile()", "IO001", 
                    FatalException, "Cannot create SimPhotons directory");
    }

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
    
    std::filesystem::path fullPath = simPhotonsDir / std::string(fileName);
    
    G4cout << "Opening output file: " << fullPath << G4endl;
    
    dataFile.open(fullPath);
    
    if (!dataFile.is_open()) {
        G4cerr << "ERROR: Failed to open file: " << fullPath << G4endl;
        G4Exception("EventProcessor::openOutputFile()", "IO002", 
                    FatalException, "Cannot open output file");
    }
    
    dataFile << "id,parent_id,neutron_id,pulse_id,x,y,z,dx,dy,dz,toa,wavelength,"
             << "parentName,px,py,pz,parentEnergy,nx,ny,nz,neutronEnergy\n";
}

void EventProcessor::openTriggerFile() {
    if (triggerFile.is_open()) triggerFile.close();

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path triggerDir = currentPath / "TriggerTimes";
    
    try {
        std::filesystem::create_directories(triggerDir);
        G4cout << "Created/verified directory: " << triggerDir << G4endl;
    } catch (const std::filesystem::filesystem_error& e) {
        G4cerr << "ERROR: Failed to create directory " << triggerDir << ": " << e.what() << G4endl;
        G4Exception("EventProcessor::openTriggerFile()", "IO003", 
                    FatalException, "Cannot create TriggerTimes directory");
    }

    G4String fileName = "trigger_data";
    if (Sim::batchSize > 0) {
        fileName += "_" + std::to_string(batchCount) + ".csv";
    } else {
        fileName += ".csv";
    }

    std::filesystem::path fullPath = triggerDir / std::string(fileName);
    
    G4cout << "Opening trigger file: " << fullPath << G4endl;
    
    triggerFile.open(fullPath);
    
    if (!triggerFile.is_open()) {
        G4cerr << "ERROR: Failed to open trigger file: " << fullPath << G4endl;
        G4Exception("EventProcessor::openTriggerFile()", "IO004", 
                    FatalException, "Cannot open trigger file");
    }
    
    triggerFile << "pulse_id,trigger_time_ns\n";
}

void EventProcessor::writeTriggerData(G4double triggerTime, G4int pulseId) {
    triggerFile << pulseId << "," << triggerTime << "\n";
    triggerFile.flush();
}

void EventProcessor::writeData() {
    for (const auto& p : photons) {
        dataFile << p.id << "," 
                 << p.parentId << "," 
                 << p.neutronId << ","
                 << p.pulseId << "," // New pulse_id column
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