#include "LumaCamMessenger.hh"
#include "SimConfig.hh"
#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

LumaCamMessenger::LumaCamMessenger(G4String* filename, G4LogicalVolume* sampleLogVolume, 
                                   G4LogicalVolume* scintLogVolume, G4int batch)
 : csvFilename(filename), sampleLog(sampleLogVolume), scintLog(scintLogVolume),
   batchSize(batch), matBuilder(new MaterialBuilder()) {
    
    messenger = new G4GenericMessenger(this, "/lumacam/", "lumacam control commands");

    G4cout << "LumaCamMessenger: Initializing with csvFilename=" 
           << (csvFilename ? *csvFilename : "null")
           << ", sampleLog=" << (sampleLog ? sampleLog->GetName() : "null")
           << ", scintLog=" << (scintLog ? scintLog->GetName() : "null")
           << ", batchSize=" << batchSize << G4endl;

    if (csvFilename) {
        messenger->DeclareProperty("csvFilename", *csvFilename)
            .SetGuidance("Set the CSV filename")
            .SetParameterName("filename", false)
            .SetDefaultValue("sim_data.csv");
    }

    if (sampleLog) {
        messenger->DeclareMethod("sampleMaterial", &LumaCamMessenger::SetMaterial)
            .SetGuidance("Set the material of the sample_log")
            .SetParameterName("material", false)
            .SetDefaultValue("G4_GRAPHITE");
    } else {
        G4cerr << "WARNING: sampleLog is nullptr, /lumacam/sampleMaterial command will not be available" << G4endl;
    }

    messenger->DeclareMethod("scintMaterial", &LumaCamMessenger::SetScintillatorMaterial)
        .SetGuidance("Set the scintillator material (EJ200, GS20 or LYSO)")
        .SetParameterName("material", false)
        .SetDefaultValue("EJ200");

    if (!scintLog) {
        G4cerr << "WARNING: scintLog is nullptr, /lumacam/scintMaterial command will fail until scintLog is set" << G4endl;
    }

    messenger->DeclareMethod("ScintThickness", &LumaCamMessenger::SetScintThickness)
        .SetGuidance("Set the scintillator half-thickness in cm")
        .SetParameterName("thickness", false)
        .SetDefaultValue("1.0");

    messenger->DeclareMethod("SampleThickness", &LumaCamMessenger::SetSampleThickness)
        .SetGuidance("Set the sample half-thickness in cm")
        .SetParameterName("thickness", false)
        .SetDefaultValue("3.75");

    messenger->DeclareProperty("batchSize", batchSize)
        .SetGuidance("Set the number of events per CSV file (0 for single file)")
        .SetParameterName("size", false)
        .SetDefaultValue("10000");

    messenger->DeclarePropertyWithUnit("tmin", "ns", Sim::TMIN)
        .SetGuidance("Set minimum emission time (ns)")
        .SetParameterName("tmin", false)
        .SetDefaultValue("0.0");

    messenger->DeclarePropertyWithUnit("tmax", "ns", Sim::TMAX)
        .SetGuidance("Set maximum emission time (ns). If > tmin, uniform distribution is applied.")
        .SetParameterName("tmax", false)
        .SetDefaultValue("0.0");

    // New commands for flux and frequency
    messenger->DeclareMethod("flux", &LumaCamMessenger::SetFlux)
        .SetGuidance("Set neutron flux in n/cm²/s")
        .SetParameterName("flux", false)
        .SetDefaultValue("0.0");

    messenger->DeclareMethod("freq", &LumaCamMessenger::SetFrequency)
        .SetGuidance("Set pulse frequency in Hz")
        .SetParameterName("freq", false)
        .SetDefaultValue("0.0");
}

LumaCamMessenger::~LumaCamMessenger() {
    delete messenger;
    delete matBuilder;
}

void LumaCamMessenger::SetMaterial(const G4String& materialName) {
    if (!sampleLog) {
        G4cerr << "ERROR: sampleLog is nullptr, cannot set material to " << materialName << G4endl;
        return;
    }
    
    G4NistManager* nistManager = G4NistManager::Instance();
    G4Material* material = nistManager->FindOrBuildMaterial(materialName);
    
    if (material) {
        G4cout << "Current sample material: " 
               << sampleLog->GetMaterial()->GetName() << G4endl;
        sampleLog->SetMaterial(material);
        G4cout << "Sample material set to: " << materialName 
               << ", Confirmed material: " 
               << sampleLog->GetMaterial()->GetName() << G4endl;
    } else {
        G4cerr << "Material " << materialName << " not found!" << G4endl;
        G4cout << "Available NIST materials:" << G4endl;
        const std::vector<G4String>& materialNames = nistManager->GetNistMaterialNames();
        for (const auto& name : materialNames) {
            G4cout << name << G4endl;
        }
    }
}

void LumaCamMessenger::SetScintillatorMaterial(const G4String& materialName) {
    if (!scintLog) {
        G4cerr << "ERROR: scintLog is nullptr, cannot set scintillator material to " << materialName << G4endl;
        return;
    }
    
    G4cout << "Setting scintillator material to: " << materialName << G4endl;
    matBuilder->setScintillatorType(materialName);
    G4Material* material = matBuilder->getScintillator();
    
    if (material) {
        G4cout << "Current scintillator material: " 
               << scintLog->GetMaterial()->GetName() << G4endl;
        scintLog->SetMaterial(material);
        G4cout << "Scintillator material set to: " << materialName 
               << ", Confirmed material: " 
               << scintLog->GetMaterial()->GetName() << G4endl;
    } else {
        G4cerr << "Scintillator material " << materialName << " not found!" << G4endl;
        G4cout << "Available scintillator materials: EJ200, GS20, LYSO" << G4endl;
    }
}

void LumaCamMessenger::SetScintThickness(G4double thickness) {
    Sim::SetScintThickness(thickness * cm);
    G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager && runManager->GetUserDetectorConstruction()) {
        G4VUserDetectorConstruction* detector = const_cast<G4VUserDetectorConstruction*>(
            runManager->GetUserDetectorConstruction());
        runManager->DefineWorldVolume(detector->Construct());
        runManager->GeometryHasBeenModified();
        G4cout << "Geometry rebuilt with new scintillator half-thickness: " << thickness << " cm" << G4endl;
    } else {
        G4cerr << "ERROR: RunManager or DetectorConstruction is nullptr, cannot rebuild geometry" << G4endl;
    }
}

void LumaCamMessenger::SetSampleThickness(G4double thickness) {
    Sim::SetSampleThickness(thickness * cm);
    G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager && runManager->GetUserDetectorConstruction()) {
        G4VUserDetectorConstruction* detector = const_cast<G4VUserDetectorConstruction*>(
            runManager->GetUserDetectorConstruction());
        runManager->DefineWorldVolume(detector->Construct());
        runManager->GeometryHasBeenModified();
        G4cout << "Geometry rebuilt with new sample half-thickness: " << thickness << " cm" << G4endl;
    } else {
        G4cerr << "ERROR: RunManager or DetectorConstruction is nullptr, cannot rebuild geometry" << G4endl;
    }
}

void LumaCamMessenger::SetFlux(G4double flux) {
    if (flux >= 0) {
        Sim::FLUX = flux;
        G4cout << "Neutron flux set to: " << flux << " n/cm²/s" << G4endl;
    } else {
        G4cerr << "ERROR: Neutron flux must be non-negative!" << G4endl;
    }
}

void LumaCamMessenger::SetFrequency(G4double freq) {
    if (freq >= 0) {
        Sim::FREQ = freq;
        G4cout << "Pulse frequency set to: " << freq / 1000 << " kHz" << G4endl;
    } else {
        G4cerr << "ERROR: Pulse frequency must be non-negative!" << G4endl;
    }
}