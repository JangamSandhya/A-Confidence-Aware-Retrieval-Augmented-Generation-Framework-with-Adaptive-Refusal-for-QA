

import json
import logging
import os
import re
from typing import List, Dict

from config import RAGConfig

logger = logging.getLogger(__name__)


PUBMED_SAMPLE = [
    {
        "id": "pm001",
        "title": "Metformin and type 2 diabetes mellitus",
        "text": (
            "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus. "
            "It reduces hepatic glucose production and improves peripheral insulin sensitivity. "
            "Common adverse effects include gastrointestinal discomfort, nausea, and diarrhoea. "
            "Lactic acidosis is a rare but serious complication, particularly in patients with renal impairment. "
            "Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m²."
        ),
    },
    {
        "id": "pm002",
        "title": "SARS-CoV-2 spike protein and ACE2 receptor binding",
        "text": (
            "SARS-CoV-2 infects human cells primarily through the binding of its spike (S) protein "
            "to angiotensin-converting enzyme 2 (ACE2). The receptor-binding domain (RBD) of the S1 "
            "subunit mediates high-affinity attachment to ACE2. Structural studies show that the RBD "
            "undergoes conformational changes between 'up' and 'down' states, with the 'up' state being "
            "the ACE2-accessible form. Mutations such as N501Y and E484K enhance ACE2 binding affinity."
        ),
    },
    {
        "id": "pm003",
        "title": "Hypertension management guidelines",
        "text": (
            "The 2023 ESC/ESH guidelines recommend a target blood pressure of below 130/80 mmHg for most "
            "adults with hypertension. First-line agents include ACE inhibitors, angiotensin receptor "
            "blockers (ARBs), calcium channel blockers, and thiazide diuretics. Beta-blockers are preferred "
            "in patients with concurrent heart failure or atrial fibrillation. Lifestyle modifications "
            "including dietary sodium restriction, regular aerobic exercise, and weight loss are integral "
            "components of hypertension management."
        ),
    },
    {
        "id": "pm004",
        "title": "mRNA vaccines: mechanism and immunogenicity",
        "text": (
            "mRNA vaccines encode the antigen of interest and are delivered via lipid nanoparticles (LNPs). "
            "Upon cellular uptake, ribosomes translate the mRNA into the target protein, which triggers "
            "both humoral and cellular immune responses. Unlike traditional vaccines, mRNA vaccines do not "
            "require live pathogens and can be manufactured rapidly. BNT162b2 (Pfizer-BioNTech) and "
            "mRNA-1273 (Moderna) demonstrated >90% efficacy against the ancestral SARS-CoV-2 strain."
        ),
    },
    {
        "id": "pm005",
        "title": "Alzheimer's disease: amyloid hypothesis",
        "text": (
            "The amyloid cascade hypothesis posits that abnormal accumulation of amyloid-beta (Aβ) peptides "
            "initiates a pathological cascade leading to Alzheimer's disease (AD). Aβ is produced by "
            "sequential cleavage of amyloid precursor protein (APP) by β- and γ-secretases. Oligomeric Aβ "
            "species are considered particularly neurotoxic. Tau hyperphosphorylation and neurofibrillary "
            "tangle formation are downstream events in AD pathology. Lecanemab, an anti-amyloid monoclonal "
            "antibody, received FDA accelerated approval in 2023."
        ),
    },
    {
        "id": "pm006",
        "title": "CRISPR-Cas9 gene editing",
        "text": (
            "CRISPR-Cas9 is a prokaryote-derived genome editing tool in which a guide RNA (gRNA) directs "
            "the Cas9 endonuclease to a specific genomic locus where it introduces a double-strand break (DSB). "
            "DSBs are repaired by non-homologous end joining (NHEJ), which often causes insertions or deletions "
            "(indels), or by homology-directed repair (HDR) in the presence of a repair template. Off-target "
            "cleavage remains a safety concern in therapeutic applications. Base editing and prime editing are "
            "refined variants that minimise DSBs."
        ),
    },
    {
        "id": "pm007",
        "title": "Atrial fibrillation: pathophysiology and anticoagulation",
        "text": (
            "Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia, characterised by "
            "disorganised atrial electrical activity. Risk factors include hypertension, heart failure, "
            "valvular disease, obesity, and sleep apnoea. AF significantly increases stroke risk due to "
            "stasis-mediated thrombus formation in the left atrial appendage. Oral anticoagulation with "
            "direct oral anticoagulants (DOACs) such as apixaban or rivaroxaban is recommended for most "
            "patients with non-valvular AF and a CHA₂DS₂-VASc score ≥2."
        ),
    },
    {
        "id": "pm008",
        "title": "Checkpoint inhibitors in oncology",
        "text": (
            "Immune checkpoint inhibitors (ICIs) target inhibitory receptors such as PD-1, PD-L1, and CTLA-4 "
            "to restore anti-tumour T-cell activity. Pembrolizumab (anti-PD-1) and nivolumab are approved "
            "for multiple malignancies including melanoma, non-small cell lung cancer, and renal cell carcinoma. "
            "Immune-related adverse events (irAEs), including colitis, pneumonitis, and endocrinopathies, "
            "are managed with immunosuppression. Tumour mutational burden (TMB) and PD-L1 expression are "
            "predictive biomarkers for ICI response."
        ),
    },
    {
        "id": "pm009",
        "title": "Sepsis: definition and management",
        "text": (
            "Sepsis-3 defines sepsis as life-threatening organ dysfunction caused by a dysregulated host "
            "response to infection, identified by a SOFA score increase of ≥2. Septic shock is a subset "
            "requiring vasopressors and having a lactate >2 mmol/L despite adequate fluid resuscitation. "
            "The Surviving Sepsis Campaign recommends blood cultures before antibiotics, broad-spectrum "
            "antibiotic administration within one hour of recognition, and 30 mL/kg crystalloid resuscitation. "
            "Norepinephrine is the first-choice vasopressor."
        ),
    },
    {
        "id": "pm010",
        "title": "Type 1 diabetes and insulin therapy",
        "text": (
            "Type 1 diabetes mellitus (T1DM) results from autoimmune destruction of pancreatic beta cells, "
            "leading to absolute insulin deficiency. Management relies on intensive insulin therapy using "
            "basal-bolus regimens or continuous subcutaneous insulin infusion (CSII). Continuous glucose "
            "monitoring (CGM) systems improve glycaemic control and reduce hypoglycaemia. Hybrid closed-loop "
            "systems (artificial pancreas) automatically adjust basal insulin delivery based on CGM readings. "
            "HbA1c targets of below 7% are recommended to reduce microvascular complications."
        ),
    },
    {
        "id": "pm011",
        "title": "Acute myocardial infarction: reperfusion therapy",
        "text": (
            "ST-elevation myocardial infarction (STEMI) requires urgent reperfusion, preferably by primary "
            "percutaneous coronary intervention (PCI) within 90 minutes of first medical contact. Fibrinolytic "
            "therapy with alteplase or tenecteplase is an alternative when PCI is unavailable within 120 minutes. "
            "Dual antiplatelet therapy with aspirin and a P2Y12 inhibitor (ticagrelor or prasugrel) reduces "
            "recurrent ischaemic events. Cardiac rehabilitation improves outcomes and quality of life "
            "post-MI."
        ),
    },
    {
        "id": "pm012",
        "title": "Gut microbiome and metabolic disease",
        "text": (
            "The gut microbiome comprises trillions of microorganisms that play pivotal roles in metabolism, "
            "immunity, and host-pathogen defence. Dysbiosis, characterised by reduced microbial diversity and "
            "altered compositional ratios, has been associated with obesity, type 2 diabetes, and "
            "non-alcoholic fatty liver disease. Short-chain fatty acids (SCFAs) produced by bacterial "
            "fermentation of dietary fibre improve insulin sensitivity and promote gut barrier integrity. "
            "Faecal microbiota transplantation (FMT) is an emerging therapeutic strategy."
        ),
    },
    {
        "id": "pm013",
        "title": "Parkinson's disease: dopaminergic pathways",
        "text": (
            "Parkinson's disease (PD) is characterised by the progressive loss of dopaminergic neurons in "
            "the substantia nigra pars compacta and the accumulation of alpha-synuclein aggregates (Lewy bodies). "
            "Motor features include resting tremor, bradykinesia, rigidity, and postural instability. "
            "Levodopa combined with a dopa-decarboxylase inhibitor (carbidopa) remains the gold standard "
            "pharmacological treatment. Dopamine agonists (pramipexole, ropinirole) and MAO-B inhibitors "
            "are used as adjunct or early monotherapy."
        ),
    },
    {
        "id": "pm014",
        "title": "BRCA1/2 mutations and hereditary breast cancer",
        "text": (
            "Germline mutations in BRCA1 and BRCA2 confer significantly elevated lifetime risks of breast "
            "and ovarian cancer: approximately 72% and 44% for breast cancer in BRCA1 carriers respectively. "
            "Both genes encode tumour suppressor proteins involved in homologous recombination DNA repair. "
            "Loss of BRCA function renders tumours sensitive to PARP inhibitors (e.g., olaparib, niraparib) "
            "due to synthetic lethality. Risk-reducing bilateral salpingo-oophorectomy is recommended for "
            "BRCA1 carriers by age 35-40."
        ),
    },
    {
        "id": "pm015",
        "title": "Chronic obstructive pulmonary disease: GOLD guidelines",
        "text": (
            "Chronic obstructive pulmonary disease (COPD) is defined by persistent airflow limitation "
            "and post-bronchodilator FEV1/FVC ratio below 0.70. The GOLD 2023 guidelines classify COPD "
            "severity into grades 1-4 based on FEV1% predicted. Inhaled bronchodilators (LABA, LAMA) "
            "are the cornerstone of pharmacological management. Inhaled corticosteroids are added for "
            "patients with frequent exacerbations and elevated eosinophil counts. Smoking cessation is "
            "the most effective intervention to slow disease progression."
        ),
    },
    {
        "id": "pm016",
        "title": "CAR-T cell therapy in haematological malignancies",
        "text": (
            "Chimeric antigen receptor T-cell (CAR-T) therapy involves engineering autologous T cells to "
            "express receptors targeting tumour-specific antigens. Axicabtagene ciloleucel targets CD19 and "
            "is approved for relapsed/refractory diffuse large B-cell lymphoma. Cytokine release syndrome "
            "(CRS) and immune effector cell-associated neurotoxicity syndrome (ICANS) are major adverse "
            "effects managed with tocilizumab and corticosteroids. Allogeneic 'off-the-shelf' CAR-T products "
            "are under active clinical development."
        ),
    },
    {
        "id": "pm017",
        "title": "Stroke: thrombolysis and thrombectomy",
        "text": (
            "Acute ischaemic stroke is caused by occlusion of a cerebral artery, resulting in loss of "
            "neurological function. Intravenous alteplase (tPA) administered within 4.5 hours of symptom "
            "onset significantly improves functional outcomes. Mechanical thrombectomy is indicated for "
            "large vessel occlusion up to 24 hours in selected patients. The NIHSS scale quantifies "
            "neurological deficits, while ASPECTS scores infarct extent on CT. Secondary prevention "
            "includes antiplatelet therapy, statin treatment, and blood pressure control."
        ),
    },
    {
        "id": "pm018",
        "title": "RNA interference: siRNA mechanisms",
        "text": (
            "RNA interference (RNAi) is a conserved eukaryotic mechanism for post-transcriptional gene "
            "silencing mediated by small interfering RNAs (siRNAs). Double-stranded siRNAs are processed "
            "by Dicer into 21-23 nucleotide duplexes that are loaded into the RNA-induced silencing complex "
            "(RISC). The antisense strand guides RISC to complementary mRNA targets, which are cleaved by "
            "the Argonaute-2 (AGO2) protein. Therapeutic siRNAs such as inclisiran (targeting PCSK9) have "
            "achieved clinical approval."
        ),
    },
    {
        "id": "pm019",
        "title": "Inflammatory bowel disease: pathogenesis",
        "text": (
            "Inflammatory bowel disease (IBD) encompasses Crohn's disease and ulcerative colitis, both "
            "driven by dysregulated mucosal immune responses in genetically susceptible individuals. Crohn's "
            "disease can affect any segment of the gastrointestinal tract with transmural inflammation, "
            "while ulcerative colitis is confined to the colon and mucosa. Anti-TNF biologics (infliximab, "
            "adalimumab) are effective for moderate-to-severe disease. Vedolizumab targets gut-selective "
            "α4β7 integrin, and ustekinumab inhibits IL-12/IL-23 pathways."
        ),
    },
    {
        "id": "pm020",
        "title": "Heart failure: BNP and management",
        "text": (
            "Heart failure (HF) affects over 64 million people worldwide and is categorised by ejection "
            "fraction: HFrEF (≤40%), HFmrEF (41-49%), and HFpEF (≥50%). B-type natriuretic peptide (BNP) "
            "and NT-proBNP are key diagnostic biomarkers. Guideline-directed medical therapy for HFrEF "
            "includes ACE inhibitors/ARBs/ARNIs, beta-blockers, MRAs, and SGLT2 inhibitors. Cardiac "
            "resynchronisation therapy (CRT) is indicated in symptomatic HFrEF with LBBB and QRS >150 ms."
        ),
    },
    {
        "id": "pm021",
        "title": "Kidney disease: CKD staging and progression",
        "text": (
            "Chronic kidney disease (CKD) is staged by GFR and albuminuria according to KDIGO guidelines. "
            "Major causes include diabetic nephropathy, hypertensive nephrosclerosis, and glomerulonephritis. "
            "RAAS blockade with ACE inhibitors or ARBs reduces proteinuria and slows CKD progression. "
            "SGLT2 inhibitors have demonstrated kidney-protective effects independent of glucose control "
            "in both diabetic and non-diabetic CKD. Renal replacement therapy (dialysis or transplantation) "
            "is required in end-stage kidney disease."
        ),
    },
    {
        "id": "pm022",
        "title": "Lung cancer: targeted therapies",
        "text": (
            "Non-small cell lung cancer (NSCLC) harbouring EGFR activating mutations (exon 19 deletions, "
            "L858R) responds to EGFR tyrosine kinase inhibitors (TKIs). Osimertinib (third-generation TKI) "
            "is the preferred first-line agent and is active against the T790M resistance mutation. ALK "
            "rearrangements occur in ~5% of NSCLC and are sensitive to alectinib or brigatinib. KRAS G12C "
            "mutations can be targeted by sotorasib or adagrasib. Comprehensive molecular profiling is "
            "recommended for all advanced NSCLC."
        ),
    },
    {
        "id": "pm023",
        "title": "Rheumatoid arthritis: JAK inhibitors",
        "text": (
            "Rheumatoid arthritis (RA) is a systemic autoimmune disease driven by synovial inflammation, "
            "mediated in part by cytokines signalling through Janus kinase (JAK) pathways. JAK inhibitors "
            "including tofacitinib, baricitinib, and upadacitinib selectively block JAK1/JAK3 or JAK1/JAK2. "
            "They are effective in methotrexate-inadequate responders. Safety signals include increased risk "
            "of herpes zoster, venous thromboembolism, and major adverse cardiovascular events (MACE) in "
            "high-risk patients. Regular monitoring is recommended."
        ),
    },
    {
        "id": "pm024",
        "title": "Multiple sclerosis: disease-modifying therapies",
        "text": (
            "Multiple sclerosis (MS) is a demyelinating autoimmune disease of the CNS. High-efficacy "
            "disease-modifying therapies (DMTs) such as natalizumab, ocrelizumab, and cladribine have "
            "substantially reduced relapse rates in relapsing-remitting MS. Ocrelizumab is the only "
            "approved therapy for primary progressive MS. Progressive multifocal leukoencephalopathy (PML) "
            "caused by JC virus reactivation is a serious risk with natalizumab, especially in JC antibody "
            "positive patients with high index values."
        ),
    },
    {
        "id": "pm025",
        "title": "Thyroid cancer: molecular classification",
        "text": (
            "Thyroid cancer encompasses papillary, follicular, medullary, and anaplastic subtypes. Papillary "
            "thyroid carcinoma (PTC) is the most common and frequently harbours BRAF V600E mutations. "
            "Medullary thyroid carcinoma arises from parafollicular C cells and may be associated with "
            "RET mutations in the context of MEN2 syndromes. Anaplastic thyroid cancer is the most "
            "aggressive variant, with BRAF/MEK inhibitor combinations showing activity in BRAF V600E "
            "positive cases. Radioactive iodine ablation is used for differentiated thyroid cancers."
        ),
    },
    {
        "id": "pm026",
        "title": "Obesity pharmacotherapy",
        "text": (
            "GLP-1 receptor agonists such as semaglutide and liraglutide are now approved for chronic "
            "weight management. Tirzepatide, a dual GIP/GLP-1 receptor agonist, demonstrated the largest "
            "mean weight reduction (~22%) in the SURMOUNT-1 trial. Orlistat inhibits pancreatic lipase "
            "and reduces dietary fat absorption. Combination phentermine/topiramate (Qsymia) and "
            "naltrexone/bupropion (Contrave) are also approved. Bariatric surgery remains the most "
            "effective long-term intervention for severe obesity."
        ),
    },
    {
        "id": "pm027",
        "title": "Neonatal sepsis: diagnosis and treatment",
        "text": (
            "Neonatal sepsis remains a leading cause of neonatal morbidity and mortality globally. "
            "Early-onset sepsis (EOS, <72 h) is predominantly caused by Group B Streptococcus and "
            "E. coli. Empirical treatment with ampicillin and gentamicin covers common pathogens. "
            "Blood culture remains the gold standard for diagnosis, though its sensitivity is limited. "
            "Serum procalcitonin and CRP are adjunctive biomarkers. Late-onset sepsis is increasingly "
            "caused by coagulase-negative staphylococci in NICU settings."
        ),
    },
    {
        "id": "pm028",
        "title": "Haemophilia: gene therapy advances",
        "text": (
            "Haemophilia A and B result from deficiencies of clotting factors VIII and IX respectively. "
            "Traditional management involves prophylactic factor replacement infusions. Extended half-life "
            "products (EHL) reduce infusion frequency. Emicizumab, a bispecific antibody mimicking factor "
            "VIII function, is effective for haemophilia A with or without inhibitors. Valoctocogene "
            "roxaparvovec (AAV5-FVIII) gene therapy achieved sustained factor VIII expression and "
            "significantly reduced bleeding rates in phase III trials."
        ),
    },
    {
        "id": "pm029",
        "title": "Antibiotic resistance mechanisms",
        "text": (
            "Antimicrobial resistance (AMR) arises through multiple mechanisms: enzymatic drug inactivation "
            "(e.g., beta-lactamases including ESBL and carbapenemases), target site modification (e.g., "
            "PBP2a in MRSA), reduced outer membrane permeability, and efflux pump overexpression. Horizontal "
            "gene transfer via plasmids accelerates AMR spread across species. Carbapenem-resistant "
            "Enterobacterales (CRE) and MRSA are WHO priority pathogens. Phage therapy and novel beta-lactam/"
            "beta-lactamase inhibitor combinations represent emerging therapeutic strategies."
        ),
    },
    {
        "id": "pm030",
        "title": "Stem cell therapy in regenerative medicine",
        "text": (
            "Induced pluripotent stem cells (iPSCs) are generated by reprogramming somatic cells with "
            "Yamanaka factors (Oct4, Sox2, Klf4, c-Myc). iPSC-derived cardiomyocytes and neural progenitors "
            "hold promise for cardiac and neurological repair. Mesenchymal stem cells (MSCs) exert "
            "immunomodulatory effects via paracrine signalling. Umbilical cord blood transplantation "
            "provides an alternative source of haematopoietic stem cells. Tumorigenicity and immune "
            "rejection remain key safety concerns in stem cell therapy."
        ),
    },
    {
        "id": "pm031",
        "title": "Pharmacokinetics: drug absorption and distribution",
        "text": (
            "Pharmacokinetics (PK) describes the time course of drug absorption, distribution, metabolism, "
            "and elimination (ADME). Bioavailability (F) reflects the fraction of an administered dose "
            "reaching systemic circulation. Volume of distribution (Vd) indicates the extent of drug "
            "distribution to tissues. Highly lipophilic drugs have large Vd values. Plasma protein binding, "
            "primarily to albumin and alpha-1 acid glycoprotein, affects the free drug fraction. "
            "Cytochrome P450 enzymes (CYP3A4, CYP2D6) are central to hepatic drug metabolism."
        ),
    },
    {
        "id": "pm032",
        "title": "Depression: neurobiology and pharmacotherapy",
        "text": (
            "Major depressive disorder (MDD) is associated with dysregulation of monoamine neurotransmission "
            "involving serotonin, norepinephrine, and dopamine. Selective serotonin reuptake inhibitors "
            "(SSRIs) are first-line pharmacotherapy due to their favourable safety profile. "
            "Serotonin-norepinephrine reuptake inhibitors (SNRIs), mirtazapine, and bupropion are "
            "alternatives. Esketamine (intranasal) provides rapid antidepressant effects in treatment-"
            "resistant depression. Neuroinflammation, HPA axis dysregulation, and reduced neuroplasticity "
            "are implicated in MDD pathophysiology."
        ),
    },
]


class Document:

    def __init__(self, doc_id: str, title: str, text: str):
        self.id = doc_id
        self.title = title
        self.text = text

    def to_dict(self) -> Dict:
        return {"id": self.id, "title": self.title, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        return cls(d.get("id", ""), d.get("title", ""), d.get("text", ""))

    def __repr__(self):
        return f"Document(id={self.id!r}, title={self.title!r})"


class CorpusLoader:

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def load(self) -> List[Document]:
        if self.cfg.corpus_source == "pubmed_sample":
            docs = self._load_sample()
        elif self.cfg.corpus_source == "json_file":
            docs = self._load_json()
        elif self.cfg.corpus_source == "txt_dir":
            docs = self._load_txt_dir()
        else:
            raise ValueError(f"Unknown corpus_source: {self.cfg.corpus_source}")

        docs = [self._preprocess(d) for d in docs]
        if self.cfg.max_documents:
            docs = docs[: self.cfg.max_documents]
        logger.info(f"Corpus loaded: {len(docs)} documents.")
        return docs


    def _load_sample(self) -> List[Document]:
        return [Document(d["id"], d["title"], d["text"]) for d in PUBMED_SAMPLE]

    def _load_json(self) -> List[Document]:
        path = self.cfg.corpus_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Corpus JSON not found: {path}")
        with open(path) as f:
            raw = json.load(f)
        return [Document(d.get("id", str(i)), d.get("title", ""), d["text"])
                for i, d in enumerate(raw)]

    def _load_txt_dir(self) -> List[Document]:
        path = self.cfg.corpus_path
        if not path or not os.path.isdir(path):
            raise NotADirectoryError(f"Corpus directory not found: {path}")
        docs = []
        for i, fname in enumerate(sorted(os.listdir(path))):
            if fname.endswith(".txt"):
                fpath = os.path.join(path, fname)
                with open(fpath) as f:
                    text = f.read()
                docs.append(Document(str(i), fname.replace(".txt", ""), text))
        return docs


    def _preprocess(self, doc: Document) -> Document:
        text = doc.text
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
        doc.text = text
        return doc
