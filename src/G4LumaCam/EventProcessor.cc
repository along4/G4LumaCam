#include "EventProcessor.hh"
#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

/**
 * Constructor for the EventProcessor
 * @param name The name of the sensitive detector
 * @param gen Pointer to the particle generator
 */
EventProcessor::EventProcessor(const G4String& name, ParticleGenerator* gen) 
    : G4VSensitiveDetector(name), neutronCount(-1), batchCount(0), eventCount(0), particleGen(gen), neutronRecorded(false) {
    resetData();
}

/**
 * Destructor for the EventProcessor
 * Closes the data file if it's open
 */
EventProcessor::~EventProcessor() {
    if (dataFile.is_open()) dataFile.close();
}

/**
 * Initialize the sensitive detector for the current event
 * @param hce The hit collection of this event
 */
void EventProcessor::Initialize(G4HCofThisEvent*) {
    resetData();
}

/**
 * Reset the data structures for a new event
 */
void EventProcessor::resetData() {
    photons.clear();
    tracks.clear();
    neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
    neutronEnergy = 0.;
    protonEnergy = 0.;
    lensPos[0] = lensPos[1] = 0.;
    neutronRecorded = false;
}

/**
 * Process hits in the sensitive detector
 * @param step The current step
 * @param touchable The touchable history
 * @return true if the hit is processed
 */
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
            neutronPos[0] = postPos.x(); // Position after interaction
            neutronPos[1] = postPos.y();
            neutronPos[2] = postPos.z();
            neutronEnergy = particleGen ? particleGen->getParticleEnergy() : track->GetKineticEnergy() / MeV;
            neutronCount++;
            neutronRecorded = true;
            // G4cout << "Primary particle (" << particleName << ") first interaction in ScintPhys at (" 
            //        << neutronPos[0]/cm << ", " << neutronPos[1]/cm << ", " << neutronPos[2]/cm 
            //        << ") cm, Energy: " << neutronEnergy << " MeV" << G4endl;
        }
    }

    // Track any particle in ScintPhys that could potentially create optical photons
    if (volName == "ScintPhys" && particleName != "opticalphoton") {
        // If this is a new track we haven't seen before, record it
        if (tracks.find(tid) == tracks.end()) {
            G4double energy = track->GetKineticEnergy() / MeV;
            // For secondary particles with zero energy, use the neutron energy 
            // as the source of energy for the nuclear reaction
            if (parentID != 0 && energy <= 0) {
                energy = neutronEnergy;
                // G4cout << "Secondary " << particleName << " has zero energy. Using neutron energy: " 
                //       << energy << " MeV instead." << G4endl;
            }
            tracks[tid] = {particleName, prePos.x(), prePos.y(), prePos.z(), energy, false};
            // G4cout << "Tracking particle in ScintPhys: " << particleName << ", TrackID: " << tid 
            //       << ", ParentID: " << parentID << ", Energy: " << energy << " MeV" << G4endl;
        }

        // Monitor for light production processes
        G4String processName = postStep->GetProcessDefinedStep() ? 
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName == "Scintillation" || processName == "Cerenkov") {
            // Update the position and mark as light producer
            tracks[tid].x = prePos.x();
            tracks[tid].y = prePos.y();
            tracks[tid].z = prePos.z();
            tracks[tid].isLightProducer = true;
            
            // If the particle has zero energy but is producing light, use neutron energy
            G4double currentEnergy = track->GetKineticEnergy() / MeV;
            if (currentEnergy <= 0 && tracks[tid].energy <= 0) {
                tracks[tid].energy = neutronEnergy;
                // G4cout << "Light produced by " << particleName << " with zero energy. Using neutron energy: " 
                //       << neutronEnergy << " MeV instead." << G4endl;
            }
            
            // G4cout << "Light produced by " << particleName << ", TrackID: " << tid 
            //       << " at (" << prePos.x()/cm << ", " << prePos.y()/cm << ", " << prePos.z()/cm 
            //       << ") cm, Current Energy: " << currentEnergy << " MeV, Using Energy: " 
            //       << tracks[tid].energy << " MeV" << G4endl;
        }
    }

    // Record optical photons hitting the monitor
    if (volName == "MonitorPhys" && particleName == "opticalphoton") {
        lensPos[0] = postStep->GetPosition().x() / mm + 500. * preStep->GetMomentumDirection().x();
        lensPos[1] = postStep->GetPosition().y() / mm + 500. * preStep->GetMomentumDirection().y();

        if (lensPos[0] > -27.5 && lensPos[0] < 27.5 && lensPos[1] > -27.5 && lensPos[1] < 27.5) {
            // Debug the parent relationship
            // G4cout << "Optical photon hit monitor: TrackID: " << tid << ", ParentID: " << parentID << G4endl;
            
            if (tracks.find(parentID) == tracks.end()) {
                // G4cout << "Warning: Parent TrackID " << parentID << " not found for optical photon. "
                //        << "Using neutron energy as fallback." << G4endl;
                
                // Create a record for the missing parent using neutron data
                tracks[parentID] = {"unknown", neutronPos[0], neutronPos[1], neutronPos[2], neutronEnergy, true};
            }
            
            // Check if parent has zero energy and fix it if needed
            if (tracks[parentID].energy <= 0) {
                tracks[parentID].energy = neutronEnergy;
                // G4cout << "Fixed zero energy for parent " << tracks[parentID].type 
                //        << " (ID: " << parentID << ") using neutron energy: " << neutronEnergy << " MeV" << G4endl;
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

/**
 * End of event processing
 * @param hce The hit collection of this event
 */
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

/**
 * Open the output file for writing data
 */
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

/**
 * Write collected data to the output file
 */
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