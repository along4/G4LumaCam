#include "LumaCamMessenger.hh"

LumaCamMessenger::LumaCamMessenger(G4String* filename, G4LogicalVolume* sampleLogVolume, 
                                   G4LogicalVolume* scintLogVolume, G4int batch)
 : csvFilename(filename), sampleLog(sampleLogVolume), scintLog(scintLogVolume), batchSize(batch), matBuilder(new MaterialBuilder()) {
    messenger = new G4GenericMessenger(this, "/lumacam/", "lumacam control commands");

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
    }

    if (scintLog) {
        messenger->DeclareMethod("scintMaterial", &LumaCamMessenger::SetScintillatorMaterial)
            .SetGuidance("Set the scintillator material (EJ200 or GS20)")
            .SetParameterName("material", false)
            .SetDefaultValue("EJ200");
    }

    messenger->DeclareProperty("batchSize", batchSize)
        .SetGuidance("Set the number of events per CSV file (0 for single file)")
        .SetParameterName("size", false)
        .SetDefaultValue("10000");
}

LumaCamMessenger::~LumaCamMessenger() {
    delete messenger;
    delete matBuilder;
}

void LumaCamMessenger::SetMaterial(const G4String& materialName) {
    if (!sampleLog) {
        G4cerr << "ERROR: sampleLog is nullptr!" << G4endl;
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
        G4cerr << "ERROR: scintLog is nullptr!" << G4endl;
        return;
    }
    
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
        G4cout << "Available scintillator materials: EJ200, GS20" << G4endl;
    }
}