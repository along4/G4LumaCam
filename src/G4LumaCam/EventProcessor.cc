#include "EventProcessor.hh"
#include "ParticleGenerator.hh" // Add this include
#include "SimConfig.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

EventProcessor::EventProcessor(const G4String& name, ParticleGenerator* gen) 
    : G4VSensitiveDetector(name), neutronCount(-1), batchCount(0), eventCount(0), particleGen(gen) {
    resetData();
}

EventProcessor::~EventProcessor() {
    if (dataFile.is_open()) dataFile.close();
}

void EventProcessor::Initialize(G4HCofThisEvent*) {
    resetData();
}

void EventProcessor::resetData() {
    photons.clear();
    // tracks.clear(); // Commented out as per your fix
    neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
    neutronEnergy = protonEnergy = 0.;
    lensPos[0] = lensPos[1] = 0.;
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

    if (volName == "ScintPhys" && postStep->GetProcessDefinedStep() && 
        postStep->GetProcessDefinedStep()->GetProcessName() != "Transportation" && 
        track->GetParentID() == 0) {
        neutronPos[0] = postPos.x();
        neutronPos[1] = postPos.y();
        neutronPos[2] = postPos.z();
        neutronEnergy = particleGen ? particleGen->getParticleEnergy() : track->GetKineticEnergy() / MeV; // Use lastEnergy
        protonEnergy = track->GetKineticEnergy() / MeV;
        neutronCount++;
    }

    if (volName == "ScintPhys" && particleName != "opticalphoton") {
        G4String processName = postStep->GetProcessDefinedStep() ? 
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName == "Scintillation" || processName == "Cerenkov") {
            G4int tid = track->GetTrackID();
            tracks[tid] = {particleName, prePos.x(), prePos.y(), prePos.z(), track->GetKineticEnergy()};
            // G4cout << processName << " by " << particleName << " at (" << prePos.x()/mm << ", " 
            //        << prePos.y()/mm << ", " << prePos.z()/mm << ") mm, TrackID: " << tid << G4endl;
        }
    }

    if (volName == "MonitorPhys" && particleName == "opticalphoton") {
        lensPos[0] = postStep->GetPosition().x() / mm + 500. * preStep->GetMomentumDirection().x();
        lensPos[1] = postStep->GetPosition().y() / mm + 500. * preStep->GetMomentumDirection().y();

        if (lensPos[0] > -27.5 && lensPos[0] < 27.5 && lensPos[1] > -27.5 && lensPos[1] < 27.5) {
            PhotonRecord rec;
            rec.id = track->GetTrackID();
            rec.parentId = track->GetParentID();
            rec.neutronId = neutronCount;
            rec.x = prePos.x() / mm;
            rec.y = prePos.y() / mm;
            rec.z = prePos.z() / mm;
            rec.dx = preDir.x();
            rec.dy = preDir.y();
            rec.dz = preDir.z();
            rec.timeOfArrival = static_cast<G4long>(track->GetGlobalTime() / 1.56255 / ns);
            rec.wavelength = 1240. / (track->GetTotalEnergy() / eV);
            rec.parentType = tracks[rec.parentId].type;
            rec.px = tracks[rec.parentId].x / mm;
            rec.py = tracks[rec.parentId].y / mm;
            rec.pz = tracks[rec.parentId].z / mm;
            rec.parentEnergy = protonEnergy;
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
    if (eventCount == 0 && batchCount == 0) openOutputFile();
    if (!photons.empty()) writeData();
    
    if (Sim::batchSize > 0) {
        eventCount++;
        if (eventCount >= Sim::batchSize) {
            batchCount++;
            eventCount = 0;
            openOutputFile();
        }
    }
    resetData();
}

void EventProcessor::openOutputFile() {
    if (dataFile.is_open()) dataFile.close();

    G4String fileName = Sim::outputFileName;

    // Remove .csv if it exists
    size_t csvPos = fileName.find(".csv");
    if (csvPos != G4String::npos) {
        fileName = fileName.substr(0, csvPos);
    }

    // Add batch number if batch processing is enabled
    if (Sim::batchSize > 0) {
        fileName += "_" + std::to_string(batchCount) + ".csv";
    } else {
        fileName += ".csv";
    }
    
    dataFile.open(fileName);
    dataFile << "id,parent_id,neutron_id,x,y,z,dx,dy,dz,toa,wavelength,"
             << "parentName,px,py,pz,parentEnergy,nx,ny,nz,neutronEnergy\n";
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