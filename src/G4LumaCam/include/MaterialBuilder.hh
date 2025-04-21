#ifndef MATERIAL_BUILDER_HH
#define MATERIAL_BUILDER_HH

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4MaterialPropertiesTable.hh"

class MaterialBuilder {
public:
    MaterialBuilder();
    G4Material* getVacuum() const { return vacuum; }
    G4Material* getAir() const { return air; }
    G4Material* getPVT() const { return scintMaterial; }
    G4Material* getGraphite() const { return sampleMaterial; }
    G4Material* getQuartz() const { return windowMaterial; }
    G4Material* getBlackMat() const { return absorberMaterial; }

private:
    G4Material* vacuum;
    G4Material* air;
    G4Material* scintMaterial;
    G4Material* sampleMaterial;
    G4Material* windowMaterial;
    G4Material* absorberMaterial;

    void defineElements();
    void setupMaterialProperties(G4Material* mat, const G4double* energies, 
                               const G4double* rindex, const G4double* abslength, 
                               int nEntries, const G4double* scintillation = nullptr);
};

#endif