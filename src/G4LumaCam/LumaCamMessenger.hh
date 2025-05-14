#ifndef LUMACAM_MESSENGER_HH
#define LUMACAM_MESSENGER_HH
#include "G4GenericMessenger.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4ios.hh"
#include "MaterialBuilder.hh"

class LumaCamMessenger {
public:
    LumaCamMessenger(G4String* filename = nullptr, 
                     G4LogicalVolume* sampleLogVolume = nullptr, 
                     G4LogicalVolume* scintLogVolume = nullptr,
                     G4int batch = 10000);
    ~LumaCamMessenger();
    void SetMaterial(const G4String& materialName);
    void SetScintillatorMaterial(const G4String& materialName);

private:
    G4String* csvFilename;
    G4LogicalVolume* sampleLog;
    G4LogicalVolume* scintLog;
    G4int batchSize;
    G4GenericMessenger* messenger;
    MaterialBuilder* matBuilder;
};

#endif