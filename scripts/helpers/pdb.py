# UniProt/PDB helpers
import requests 
import pandas as pd 



def gene_to_uniprot(gene_name, organism_id="9606"):  
    # 9606 = Homo sapiens
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene_name}+AND+organism_id:{organism_id}&fields=accession&format=json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to search for gene {gene_name}")
        return None
    results = response.json().get("results", [])
    if not results:
        print(f"No UniProt accession found for gene {gene_name}")
        return None
    return results[0]["primaryAccession"]

def fetch_pdb_ids(uniprot_accession):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {uniprot_accession}")
        return None, pd.DataFrame()
    response_json = response.json()
    rows = []
    for entry in response_json.get("uniProtKBCrossReferences", []):
        if entry["database"] == "PDB":
            row = {"PDB_ID": entry["id"]}
            for prop in entry.get("properties", []):
                row[prop["key"]] = prop["value"]
            rows.append(row)
    df_pdb_ids = pd.DataFrame(rows)
    pdb_id = df_pdb_ids.iloc[0]["PDB_ID"] if not df_pdb_ids.empty else None
    return pdb_id, df_pdb_ids

def gene_to_pdb(gene_name):
    accession = gene_to_uniprot(gene_name)
    if accession is None:
        return None, pd.DataFrame()
    return fetch_pdb_ids(accession)