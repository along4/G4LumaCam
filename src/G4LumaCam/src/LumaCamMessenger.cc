#include "LumaCamMessenger.hh"

LumaCamMessenger::LumaCamMessenger(G4String* filename, G4LogicalVolume* sampleLogVolume, G4int batch)
 : csvFilename(filename), sampleLog(sampleLogVolume), batchSize(batch) {
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

    messenger->DeclareProperty("batchSize", batchSize)
        .SetGuidance("Set the number of events per CSV file (0 for single file)")
        .SetParameterName("size", false)
        .SetDefaultValue("10000");
}

LumaCamMessenger::~LumaCamMessenger() {
    delete messenger;
}

// Add verbose logging in SetMaterial method
void LumaCamMessenger::SetMaterial(const G4String& materialName) {
    if (!sampleLog) {
        G4cerr << "ERROR: sampleLog is nullptr!" << G4endl;
        return;
    }
    
    G4NistManager* nistManager = G4NistManager::Instance();
    G4Material* material = nistManager->FindOrBuildMaterial(materialName);
    
    if (material) {
        // Print current material before change
        G4cout << "Current sample material: " 
               << sampleLog->GetMaterial()->GetName() << G4endl;
        
        sampleLog->SetMaterial(material);
        
        // Verify material change
        G4cout << "Sample material set to: " << materialName 
               << ", Confirmed material: " 
               << sampleLog->GetMaterial()->GetName() << G4endl;
    } else {
        G4cerr << "Material " << materialName << " not found!" << G4endl;
        
        // List available NIST materials
        G4cout << "Available NIST materials:" << G4endl;
        const std::vector<G4String>& materialNames = nistManager->GetNistMaterialNames();
        for (const auto& name : materialNames) {
            G4cout << name << G4endl;
        }
    }
}