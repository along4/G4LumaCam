#include "MaterialBuilder.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include <vector>

MaterialBuilder::MaterialBuilder() {
    G4NistManager* nist = G4NistManager::Instance();
    
    // Elements
    G4Element* H = new G4Element("H", "H", 1., 1.01 * g/mole);
    G4Element* C = new G4Element("C", "C", 6., 12.01 * g/mole);
    G4Element* N = new G4Element("N", "N", 7., 14.01 * g/mole);
    G4Element* O = new G4Element("O", "O", 8., 16.00 * g/mole);
    G4Element* Si = new G4Element("Si", "Si", 14., 28.09 * g/mole);
    
    // Vacuum
    vacuum = new G4Material("Vacuum", 1., 1.01 * g/mole, CLHEP::universe_mean_density,
        kStateGas, 2.73 * kelvin, 3.e-18 * pascal);
    
    // Air
    air = new G4Material("Air", 1.290 * mg/cm3, 2);
    air->AddElement(N, 70 * perCent);
    air->AddElement(O, 30 * perCent);
    
    G4double airEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double airRIndex[2] = {1.0, 1.0};
    G4double airAbs[2] = {100.0 * m, 100.0 * m};
    setupMaterialProperties(air, airEnergy, airRIndex, airAbs, 2);
    
    
    // Scintillator (PVT)
    scintMaterial = new G4Material("Scintillator", 1.023 * g/cm3, 2);
    scintMaterial->AddElement(C, 9);
    scintMaterial->AddElement(H, 10);

    // Number of points in the updated data
    const int nPVT = 61;

    // Convert wavelength (nm) to energy (eV) using E = hc/λ
    // Energy values in eV, calculated from wavelength data
    G4double pvtEnergy[nPVT] = {
        3.26*eV, 3.25*eV, 3.23*eV, 3.21*eV, 3.20*eV, 3.18*eV, 3.16*eV, 3.15*eV, 3.13*eV, 3.12*eV,
        3.10*eV, 3.08*eV, 3.07*eV, 3.05*eV, 3.04*eV, 3.02*eV, 3.01*eV, 2.99*eV, 2.98*eV, 2.96*eV,
        2.95*eV, 2.94*eV, 2.92*eV, 2.91*eV, 2.90*eV, 2.88*eV, 2.87*eV, 2.85*eV, 2.84*eV, 2.82*eV,
        2.81*eV, 2.80*eV, 2.79*eV, 2.77*eV, 2.76*eV, 2.75*eV, 2.74*eV, 2.73*eV, 2.72*eV, 2.70*eV,
        2.69*eV, 2.68*eV, 2.67*eV, 2.66*eV, 2.65*eV, 2.64*eV, 2.63*eV, 2.62*eV, 2.61*eV, 2.59*eV,
        2.58*eV, 2.57*eV, 2.56*eV, 2.55*eV, 2.54*eV, 2.52*eV, 2.51*eV, 2.50*eV, 2.49*eV, 2.48*eV,
        2.48*eV
    };

    // Scintillation intensity values from the provided table
    G4double pvtScint[nPVT] = {
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.002,
        0.006, 0.016, 0.044, 0.101, 0.173, 0.251, 0.348, 0.454, 0.596, 0.728,
        0.861, 0.949, 0.991, 0.999, 0.982, 0.956, 0.926, 0.890, 0.850, 0.804,
        0.755, 0.706, 0.658, 0.617, 0.582, 0.549, 0.519, 0.492, 0.468, 0.448,
        0.429, 0.410, 0.389, 0.364, 0.337, 0.306, 0.271, 0.238, 0.212, 0.191,
        0.171, 0.153, 0.137, 0.123, 0.109, 0.098, 0.087, 0.077, 0.068, 0.059,
        0.056
    };

    // Refractive index values (constant at 1.58 as in the original code)
    G4double pvtRIndex[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtRIndex[i] = 1.58;
    }

    // Absorption length values (constant at 380 cm as in the original code)
    G4double pvtAbs[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtAbs[i] = 380.*cm;
    }

    // Set material properties
    setupMaterialProperties(scintMaterial, pvtEnergy, pvtRIndex, pvtAbs, nPVT, pvtScint);
    scintMaterial->GetIonisation()->SetBirksConstant(0.126 * mm/MeV);
    
    // Graphite
    sampleMaterial = nist->FindOrBuildMaterial("G4_GRAPHITE");
    
    // Quartz Window
    windowMaterial = new G4Material("Quartz", 2.20 * g/cm3, 2);
    windowMaterial->AddElement(Si, 1);
    windowMaterial->AddElement(O, 2);
    
    G4double quartzEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double quartzRIndex[2] = {1.59, 1.59};
    G4double quartzAbs[2] = {160.0 * cm, 160.0 * cm};
    setupMaterialProperties(windowMaterial, quartzEnergy, quartzRIndex, quartzAbs, 2);
    
    // Black Material
    absorberMaterial = new G4Material("Absorber", 1.290 * mg/cm3, 2);
    absorberMaterial->AddElement(N, 70 * perCent);
    absorberMaterial->AddElement(O, 30 * perCent);
    
    G4double blackEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double blackRIndex[2] = {1.58, 1.58};
    G4double blackAbs[2] = {0.0 * mm, 0.0 * mm};
    setupMaterialProperties(absorberMaterial, blackEnergy, blackRIndex, blackAbs, 2);
}

void MaterialBuilder::setupMaterialProperties(G4Material* mat, const G4double* energies,
    const G4double* rindex, const G4double* abslength,
    int nEntries, const G4double* scintillation) {
    // Create non-const copies of the input arrays
    std::vector<G4double> energiesCopy(energies, energies + nEntries);
    std::vector<G4double> rindexCopy(rindex, rindex + nEntries);
    std::vector<G4double> abslengthCopy(abslength, abslength + nEntries);
    
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    
    // Use the non-const copies
    mpt->AddProperty("RINDEX", energiesCopy.data(), rindexCopy.data(), nEntries);
    mpt->AddProperty("ABSLENGTH", energiesCopy.data(), abslengthCopy.data(), nEntries);
    
    if (scintillation) {
        std::vector<G4double> scintillationCopy(scintillation, scintillation + nEntries);
        mpt->AddProperty("FASTCOMPONENT", energiesCopy.data(), scintillationCopy.data(), nEntries);
        mpt->AddConstProperty("SCINTILLATIONYIELD", 10000./MeV);
        mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
        mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT", 2.1 * ns);
        mpt->AddConstProperty("SCINTILLATIONRISETIME", 0.9 * ns);
        mpt->AddConstProperty("FASTTIMECONSTANT", 3.2 * ns);
        mpt->AddConstProperty("YIELDRATIO", 0.);

    }
    
    mat->SetMaterialPropertiesTable(mpt);
}