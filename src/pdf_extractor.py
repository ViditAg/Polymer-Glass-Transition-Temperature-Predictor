"""
PDF data extraction using pdfplumber for polymer datasets.
"""

import tqdm
import pandas as pd
import pdfplumber

def _merge_rows(df, nrows=5, is_head=True):
    if is_head:
        df_subset = df.head(nrows)
    else:
        df_subset = df.tail(nrows)
    new_df = df_subset.apply(lambda x: "".join(x.tolist()), axis=0)
    
    return new_df.to_frame().T


def _get_leftover_data() -> pd.DataFrame:
    """
    Returns a DataFrame with leftover data that couldn't be extracted from the PDF.
    """
    df = pd.DataFrame(
        [
            {
                "ID": "SFT_I10",
                "logTg": '2.5226',
                "Series": "T",
                "name": "diphenyl 2,3,5,6-tetrachloro-4-[2-(2,3,5,6-tetrachlorophenyl)propan-2-yl]phenyl phosphate",
                "SMILES": "CC(C)(c1c(Cl)c(Cl)cc(Cl)c1Cl)c1c(Cl)c(Cl)c(OP(=O)(Oc2ccccc2)Oc2ccccc2)c(Cl)c1Cl",
            },
            {
                "ID": "SFT_I2",
                "logTg": '2.5455',
                "Series": "P",
                "name": "methyl phenyl 4-(2-phenylpropan-2-yl)phenyl phosphate",
                "SMILES": "COP(=O)(Oc1ccccc1)Oc1ccc(cc1)C(C)(C)c1ccccc1",
            },
            {
                "ID": "SFT_I3",
                "logTg": '2.5601',
                "Series": "T",
                "name": "butyl phenyl 4-(2-phenylpropan-2-yl)phenyl phosphate",
                "SMILES": "CCCCOP(=O)(Oc1ccccc1)Oc1ccc(cc1)C(C)(C)c1ccccc1",
            },
            {
                "ID": "SFT_I4",
                "logTg": '2.548',
                "Series": "P",
                "name": "phenyl 4-(2-phenylpropan-2-yl)phenyl butylphosphonate",
                "SMILES": "CCCCP(=O)(Oc1ccccc1)Oc1ccc(cc1)C(C)(C)c1ccccc1",
            },
            {
                "ID": "SFT_I5",
                "logTg": '2.5637',
                "Series": "T",
                "name": "phenyl 4-(2-phenylpropan-2-yl)phenyl cyclohexylphosphonate", 
                "SMILES": "CC(C)(c1ccccc1)c1ccc(OP(=O)(Oc2ccccc2)C2CCCCC2)cc1",
            },
            {
                "ID": "SFT_I6",
                "logTg": '2.5707',
                "Series": "P",
                "name": "phenyl 4-(2-phenylpropan-2-yl)phenyl phenylphosphonate",
                "SMILES": "CC(C)(c1ccccc1)c1ccc(OP(=O)(Oc2ccccc2)c2ccccc2)cc1",
            },
            {
                "ID": "SFT_I9",
                "logTg": '2.5541',
                "Series": "T",
                "name": "phenyl 2,3,5,6-tetrachloro-4-[2-(2,3,5,6-tetrachlorophenyl)propan-2-yl]phenyl phenylphosphonate",
                "SMILES": "CC(C)(c1c(Cl)c(Cl)cc(Cl)c1Cl)c1c(Cl)c(Cl)c(OP(=O)(Oc2ccccc2)c2ccccc2)c(Cl)c1Cl",
            },
            {
                "ID": "SFT_II11",
                "logTg": '2.6055',
                "Series": "T",
                "name": "[1,1'-biphenyl]-4-yl phenyl methylphosphonate",
                "SMILES": "CP(=O)(Oc1ccccc1)Oc1ccc(cc1)c1ccccc1",
            },
            {
                "ID": "SFT_II12",
                "logTg": '2.6022',
                "Series": "T",
                "name": "[1,1'-biphenyl]-4-yl phenyl cyclohexylphosphonate",
                "SMILES": "O=P(Oc1ccccc1)(Oc1ccc(cc1)c1ccccc1)C1CCCCC1",
            },
            {
                "ID": "SFT_II14",
                "logTg": '2.5901',
                "Series": "P",
                "name": "[1,1'-biphenyl]-4-yl cyclohexyl phenyl phosphate",
                "SMILES": "O=P(OC1CCCCC1)(Oc1ccccc1)Oc1ccc(cc1)c1ccccc1",
            },
            {
                "ID": "SFT_III16",
                "logTg": '2.6',
                "Series": "P",
                "name": "diphenyl cyclohexylphosphonate",
                "SMILES": "O=P(Oc1ccccc1)(Oc1ccccc1)C1CCCCC1",
            },
            {
                "ID": "SFT_III18",
                "logTg": '2.5811',
                "Series": "T",
                "name": "cyclohexyl diphenyl phosphate",
                "SMILES": "O=P(OC1CCCCC1)(Oc1ccccc1)Oc1ccccc1",
            },
            {
                "ID": "SFT_IV20",
                "logTg": '2.6076',
                "Series": "T",
                "name": "4-(benzenesulfonyl)phenyl phenyl cyclohexylphosphonate",
                "SMILES": "O=P(Oc1ccccc1)(Oc1ccc(cc1)S(=O)(=O)c1cccc1)C1CCCCC1",
            },
            {
                "ID": "SFT_IV22",
                "logTg": '2.5946',
                "Series": "T",
                "name": "4-(benzenesulfonyl)phenyl cyclohexyl phenyl phosphate",
                "SMILES": "O=P(OC1CCCCC1)(Oc1ccccc1)Oc1ccc(cc1)S(=O)(=O)c1ccccc1",
            },
        ]
    )
    
    #
    df['SMI_generator'] = "Reference"
    df['Ref'] = 'X'
    
    return df


def _read_pdf(pdf_path: str, debug_mode: bool = False) -> pd.DataFrame:
    """
    Reads the PDF and extracts tables, handling headers and footers.

    Args:
        pdf_path (str): Path to the PDF file.
    """
    columns = ["ID", "logTg", "Series", "Ref", "name", "SMILES", "SMI_generator"]
    if debug_mode:
        print(f"Using columns: {columns}")

    pages_w_header_1 = [2, 4, 5, 7, 8, 13, 14, 15, 16, 19, 20, 21, 24, 25]
    pages_w_header_merge_2 = [3, 6, 10, 17, 18, 22, 23, 26, 27, 28, 29, 31, 35, 38]
    pages_w_header_merge_3 = [9, 11, 12, 30, 32, 33, 34, 36, 39]
    pages_w_header_merge_4 = [37]
    
    pages_w_footer_1 = [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 30]
    pages_w_footer_merge_2 = [23, 24, 25, 26, 27, 28, 33, 37, 38, 39]
    pages_w_footer_merge_3 = [8, 9, 10, 11, 29, 31, 32, 34, 35]
    pages_w_footer_merge_4 = [36]
    
    base_settings = {
        "vertical_strategy": "text",    # Use text alignment instead of lines
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "snap_tolerance": 1,
        "join_tolerance": 1
    }

    major_chuck_extractor_settings = base_settings.copy()
    major_chuck_extractor_settings.update({
        "horizontal_strategy": "lines",  # Use lines to catch separators
    })
    if debug_mode: print(major_chuck_extractor_settings)

    header_footer_extractor_settings = base_settings.copy()
    header_footer_extractor_settings.update({
        "horizontal_strategy": "text",   # Use text alignment instead of lines
    })

    if debug_mode: print(header_footer_extractor_settings)
    
    final_df_list = []
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")
        
        for page_num, page in tqdm.tqdm(enumerate(pdf.pages, 1), total=len(pdf.pages)):
            
            if page_num >= 40:
                print(f"Skipping page {page_num} due to extraction issues. Will copy manually later.")
                continue

            major_page_tables = page.extract_tables(table_settings=major_chuck_extractor_settings)
            if debug_mode: print(f"  Found {len(major_page_tables)} table(s)")

            major_df = pd.DataFrame(major_page_tables[0], columns=columns)

            if page_num == 1:
                major_df = major_df[1:]  # Data without header

            minor_page_tables = page.extract_tables(table_settings=header_footer_extractor_settings)
            if debug_mode: print(f"  Found {len(minor_page_tables)} table(s)")

            minor_df = pd.DataFrame(minor_page_tables[0], columns=columns)
            
            
            if page_num in pages_w_header_1:
                header_df = minor_df.head(1)  # 
            elif page_num in pages_w_header_merge_2:
                header_df = _merge_rows(minor_df,nrows=3)  
            elif page_num in pages_w_header_merge_3:
                header_df = _merge_rows(minor_df,nrows=5) 
            elif page_num in pages_w_header_merge_4:
                header_df = _merge_rows(minor_df,nrows=7)  
            else:
                header_df = pd.DataFrame(columns=columns)  # Empty DataFrame if no header to remove  
            
            if page_num in pages_w_footer_1:
                footer_df = minor_df.tail(1) 
            elif page_num in pages_w_footer_merge_2:
                footer_df = _merge_rows(minor_df, nrows=3, is_head=False)  # 
            elif page_num in pages_w_footer_merge_3:
                footer_df = _merge_rows(minor_df, nrows=5, is_head=False)  # 
            elif page_num in pages_w_footer_merge_4:
                footer_df = _merge_rows(minor_df, nrows=7, is_head=False)  #
            if debug_mode:
                print(minor_df.head(7))
                print(header_df)
                print(minor_df.tail(7))
                print(footer_df)
                print(major_df)
            
            final_df_list.append(pd.concat([header_df, major_df, footer_df]).reset_index(drop=True))
        
        return pd.concat(final_df_list).reset_index(drop=True)
        

def extract_tables_from_pdf(
        pdf_path: str,
        debug_mode: bool = False
        ) -> pd.DataFrame:
    """
    Extract tables from a PDF file using pdfplumber, handling headers and footers.

    Args:
        pdf_path (str): Path to the PDF file.
        debug_mode (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        pd.DataFrame: Combined DataFrame with extracted data.
    """


    pdf_data_df = _read_pdf(pdf_path, debug_mode=debug_mode)  

    print("Total rows extracted directly from pdf:", pdf_data_df.shape[0])

    leftover_data_df = _get_leftover_data()
    print("Total leftover rows added manually:", leftover_data_df.shape[0])

    final_df_w_leftover = pd.concat([pdf_data_df, leftover_data_df]).reset_index(drop=True)
    
    print("Total rows all together:", final_df_w_leftover.shape[0])

    print("Removing \\n from SMILES and name strings")
    final_df_w_leftover['SMILES_clean'] = final_df_w_leftover['SMILES'].str.replace('\n', '')
    final_df_w_leftover['name_clean'] = final_df_w_leftover['name'].str.replace('\n', '')

    final_df_w_leftover.rename(
        columns={
            'SMILES': 'SMILES_raw',
            'name': 'name_raw',
            'SMILES_clean': 'SMILES',
            'name_clean': 'name'
        },
            inplace=True
        )

    return final_df_w_leftover