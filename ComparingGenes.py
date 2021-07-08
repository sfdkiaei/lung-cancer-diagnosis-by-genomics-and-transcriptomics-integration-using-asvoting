# http://bioinformatics.psb.ugent.be/webtools/Venn/
import numpy as np
import pandas as pd


def getGenesCGC(filepath: str, save=True):
    """
    CGC-cancer_gene_census.csv
    All cancer related genes from CGC
    downloaded from: https://cancer.sanger.ac.uk/cosmic/download
    :param filepath:
    :return:
    """
    data = pd.read_csv(filepath)
    genes = np.unique(data['Gene Symbol'].to_numpy())
    if save:
        with open('CGC.txt', 'w') as f:
            for gene in genes:
                f.write(gene + '\n')
    return genes


def getGenesNCG(filepath: str, save=True):
    """
    NCG-All-NCG6_cancergenes.tsv
    List of 2372 cancer genes and supporting literature
    :param filepath:
    :return:
    """
    data = pd.read_csv(filepath, sep='\t')
    genes = np.unique(data['symbol'].to_numpy())
    if save:
        with open('NCG.txt', 'w') as f:
            for gene in genes:
                f.write(gene + '\n')
    return genes


def getGenesOMIM(filepath: str, keywords: list, save=True):
    """
    OMIM_Oiginal.xlsx
    :param filepath:
    :return:
    """
    genes = []
    data = pd.read_excel(filepath, skiprows=3)
    for idx, row in data.iterrows():
        try:
            phenotype = row['Phenotypes']
            for keyword in keywords:
                if keyword.lower() in phenotype.lower():
                    gene_symbols = row['Gene Symbols'].split(',')
                    genes_symbols = [gene.strip() for gene in gene_symbols]
                    genes += genes_symbols
        except:
            pass
    if save:
        with open('OMIM.txt', 'w') as f:
            for gene in genes:
                f.write(gene + '\n')
    return np.unique(np.array(genes))


OMIM_cancer_keywords = [
    'cancer',
    'melanoma',
    'carcinoma',
    'glioma',
    'lymphoma',
    'sarcoma',
    'amyloidosis',
    'astrocytoma',
    'craniopharyngioma',
    'ependymoma',
    'leukemia',
    'mesothelioma',
    'meningioma',
    'mastocytosis'
]
path_cgc = "Data/KnownDatasets/CGC-cancer_gene_census.csv"
path_ncg = "Data/KnownDatasets/NCG-All-NCG6_cancergenes.tsv"
path_omim = "Data/KnownDatasets/OMIM_Original.xlsx"

cgc = getGenesCGC(path_cgc)
ncg = getGenesNCG(path_ncg)
omim = getGenesOMIM(filepath=path_omim, keywords=OMIM_cancer_keywords)

print(cgc.size, cgc)
print(ncg.size, ncg)
print(omim.size, omim)
