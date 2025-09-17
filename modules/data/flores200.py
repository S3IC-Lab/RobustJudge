# data/flores200.py

import os
import json
from modules.data.base_loader import BaseDataset
from modules.data.registry import DatasetRegistry

@DatasetRegistry.register("flores200")
class Flores200Dataset(BaseDataset):
    """
    Implementation for the Flores200 dataset.
    Processes specific language files and generates a JSON file with id, source, and target.
    """
    def __init__(self, **kwargs):
        super(Flores200Dataset, self).__init__(**kwargs)
        self.language_data = {}
        self.total_entries = 0

    def load_data(self):
        """
        Load data by reading all relevant language files in the data directory.
        This method aggregates all lines from each language file into a dictionary with language codes as keys.
        """
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.dev'):
                lang_code = filename.split('.')[0]  # e.g., 'zho_Hans' from 'zho_Hans.dev'
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.language_data[lang_code] = [line.strip() for line in lines]
        # Verify that all languages have the same number of lines
        lengths = [len(lines) for lines in self.language_data.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All language files must have the same number of lines.")
        self.total_entries = lengths[0]
        print(f"Loaded {self.total_entries} entries from Flores200 dataset.")

    def process_data(self):
        """
        Process the loaded data to extract source and target language pairs.
        Reads 'source_lang', 'target_lang', and 'num_objects' from self.params.
        """
        source_lang = self.params.get('source_lang')
        target_lang = self.params.get('target_lang')
        num_objects = self.params.get('num_objects', 100)

        if not source_lang or not target_lang:
            raise ValueError("source_lang and target_lang must be specified for Flores200Dataset.")

        # Validate language codes
        if source_lang not in self.language_data:
            raise ValueError(f"Source language '{source_lang}' not found in the dataset.")
        if target_lang not in self.language_data:
            raise ValueError(f"Target language '{target_lang}' not found in the dataset.")

        processed = []
        for idx in range(min(num_objects, self.total_entries)):
            entry = {
                "id": idx + 1,
                "source": self.language_data[source_lang][idx],
                "target": self.language_data[target_lang][idx],
                "model_generated": None
            }
            processed.append(entry)
        self.data = processed
        print(f"Processed {len(self.data)} entries: {source_lang} -> {target_lang}")

def lan_flores_dict(lan: str):
    dict = {'Acehnese (Arabic script)': 'ace_Arab',	\
            'Acehnese (Latin script)': 'ace_Latn',	\
            'Mesopotamian Arabic': 'acm_Arab',	\
            'Ta’izzi-Adeni Arabic': 'acq_Arab',	\
            'Tunisian Arabic': 'aeb_Arab',	\
            'Afrikaans': 'afr_Latn',	\
            'South Levantine Arabic': 'ajp_Arab',	\
            'Akan': 'aka_Latn',	\
            'Amharic': 'amh_Ethi',	\
            'North Levantine Arabic': 'apc_Arab',	\
            'Modern Standard Arabic': 'arb_Arab',	\
            'Modern Standard Arabic (Romanized)': 'arb_Latn',	\
            'Najdi Arabic': 'ars_Arab',	\
            'Moroccan Arabic': 'ary_Arab',	\
            'Egyptian Arabic': 'arz_Arab',	\
            'Assamese': 'asm_Beng',	\
            'Asturian': 'ast_Latn',	\
            'Awadhi': 'awa_Deva',	\
            'Central Aymara': 'ayr_Latn',	\
            'South Azerbaijani': 'azb_Arab',	\
            'North Azerbaijani': 'azj_Latn',	\
            'Bashkir': 'bak_Cyrl',	\
            'Bambara': 'bam_Latn',	\
            'Balinese': 'ban_Latn',	\
            'Belarusian': 'bel_Cyrl',	\
            'Bemba': 'bem_Latn',	\
            'Bengali': 'ben_Beng',	\
            'Bhojpuri': 'bho_Deva',	\
            'Banjar (Arabic script)': 'bjn_Arab',	\
            'Banjar (Latin script)': 'bjn_Latn',	\
            'Standard Tibetan': 'bod_Tibt',	\
            'Bosnian': 'bos_Latn',	\
            'Buginese': 'bug_Latn',	\
            'Bulgarian': 'bul_Cyrl',	\
            'Catalan': 'cat_Latn',	\
            'Cebuano': 'ceb_Latn',	\
            'Czech': 'ces_Latn',	\
            'Chokwe': 'cjk_Latn',	\
            'Central Kurdish': 'ckb_Arab',	\
            'Crimean Tatar': 'crh_Latn',	\
            'Welsh': 'cym_Latn',	\
            'Danish': 'dan_Latn',	\
            'German': 'deu_Latn',	\
            'Southwestern Dinka': 'dik_Latn',	\
            'Dyula': 'dyu_Latn',	\
            'Dzongkha': 'dzo_Tibt',	\
            'Greek': 'ell_Grek',	\
            'English': 'eng_Latn',	\
            'Esperanto': 'epo_Latn',	\
            'Estonian': 'est_Latn',	\
            'Basque': 'eus_Latn',	\
            'Ewe': 'ewe_Latn',	\
            'Faroese': 'fao_Latn',	\
            'Fijian': 'fij_Latn',	\
            'Finnish': 'fin_Latn',	\
            'Fon': 'fon_Latn',	\
            'French': 'fra_Latn',	\
            'Friulian': 'fur_Latn',	\
            'Nigerian Fulfulde': 'fuv_Latn',	\
            'Scottish Gaelicv': 'gla_Latn',	\
            'Irish': 'gle_Latn',	\
            'Galician': 'glg_Latn',	\
            'Guarani': 'grn_Latn',	\
            'Gujarati': 'guj_Gujr',	\
            'Haitian Creole': 'hat_Latn',	\
            'Hausa': 'hau_Latn',	\
            'Hebrew': 'heb_Hebr',	\
            'Hindi': 'hin_Deva',	\
            'Chhattisgarhi': 'hne_Deva',	\
            'Croatian': 'hrv_Latn',	\
            'Hungarian': 'hun_Latn',	\
            'Armenian': 'hye_Armn',	\
            'Igbo': 'ibo_Latn',	\
            'Ilocano': 'ilo_Latn',	\
            'Indonesian': 'ind_Latn',	\
            'Icelandic': 'isl_Latn',	\
            'Italian': 'ita_Latn',	\
            'Javanese': 'jav_Latn',	\
            'Japanese': 'jpn_Jpan',	\
            'Kabyle': 'kab_Latn',	\
            'Jingpho': 'kac_Latn',	\
            'Kamba': 'kam_Latn',	\
            'Kannada': 'kan_Knda',	\
            'Kashmiri (Arabic script)': 'kas_Arab',	\
            'Kashmiri (Devanagari script)': 'kas_Deva',	\
            'Georgian': 'kat_Geor',	\
            'Central Kanuri (Arabic script)': 'knc_Arab',	\
            'Central Kanuri (Latin script)': 'knc_Latn',	\
            'Kazakh': 'kaz_Cyrl',	\
            'Kabiyè': 'kbp_Latn',	\
            'Kabuverdianu': 'kea_Latn',	\
            'Khmer': 'khm_Khmr',	\
            'Kikuyu': 'kik_Latn',	\
            'Kinyarwanda': 'kin_Latn',	\
            'Kyrgyz': 'kir_Cyrl',	\
            'Kimbundu': 'kmb_Latn',	\
            'Northern Kurdish': 'kmr_Latn',	\
            'Kikongo': 'kon_Latn',	\
            'Korean': 'kor_Hang',	\
            'Lao': 'lao_Laoo',	\
            'Ligurian': 'lij_Latn',	\
            'Limburgish': 'lim_Latn',	\
            'Lingala': 'lin_Latn',	\
            'Lithuanian': 'lit_Latn',	\
            'Lombard': 'lmo_Latn',	\
            'Latgalian': 'ltg_Latn',	\
            'Luxembourgish': 'ltz_Latn',	\
            'Luba-Kasai': 'lua_Latn',	\
            'Ganda': 'lug_Latn',	\
            'Luo': 'luo_Latn',	\
            'Mizo': 'lus_Latn',	\
            'Standard Latvian': 'lvs_Latn',	\
            'Magahi': 'mag_Deva',	\
            'Maithili': 'mai_Deva',	\
            'Malayalam': 'mal_Mlym',	\
            'Marathi': 'mar_Deva',	\
            'Minangkabau (Arabic script)': 'min_Arab',	\
            'Minangkabau (Latin script)': 'min_Latn',	\
            'Macedonian': 'mkd_Cyrl',	\
            'Plateau Malagasy': 'plt_Latn',	\
            'Maltese': 'mlt_Latn',	\
            'Meitei (Bengali script)': 'mni_Beng',	\
            'Halh Mongolian': 'khk_Cyrl',	\
            'Mossi': 'mos_Latn',	\
            'Maori': 'mri_Latn',	\
            'Burmese': 'mya_Mymr',	\
            'Dutch': 'nld_Latn',	\
            'Norwegian Nynorsk': 'nno_Latn',	\
            'Norwegian Bokmål': 'nob_Latn',	\
            'Nepali': 'npi_Deva',	\
            'Northern Sotho': 'nso_Latn',	\
            'Nuer': 'nus_Latn',	\
            'Nyanja': 'nya_Latn',	\
            'Occitan': 'oci_Latn',	\
            'West Central Oromo': 'gaz_Latn',	\
            'Odia': 'ory_Orya',	\
            'Pangasinan': 'pag_Latn',	\
            'Eastern Panjabi': 'pan_Guru',	\
            'Papiamento': 'pap_Latn',	\
            'Western Persian': 'pes_Arab',	\
            'Polish': 'pol_Latn',	\
            'Portuguese': 'por_Latn',	\
            'Dari': 'prs_Arab',	\
            'Southern Pashto': 'pbt_Arab',	\
            'Ayacucho Quechua': 'quy_Latn',	\
            'Romanian': 'ron_Latn',	\
            'Rundi': 'run_Latn',	\
            'Russian': 'rus_Cyrl',	\
            'Sango': 'sag_Latn',	\
            'Sanskrit': 'san_Deva',	\
            'Santali': 'sat_Olck',	\
            'Sicilian': 'scn_Latn',	\
            'Shan': 'shn_Mymr',	\
            'Sinhala': 'sin_Sinh',	\
            'Slovak': 'slk_Latn',	\
            'Slovenian': 'slv_Latn',	\
            'Samoan': 'smo_Latn',	\
            'Shona': 'sna_Latn',	\
            'Sindhi': 'snd_Arab',	\
            'Somali': 'som_Latn',	\
            'Southern Sotho': 'sot_Latn',	\
            'Spanish': 'spa_Latn',	\
            'Tosk Albanian': 'als_Latn',	\
            'Sardinian': 'srd_Latn',	\
            'Serbian': 'srp_Cyrl',	\
            'Swati': 'ssw_Latn',	\
            'Sundanese': 'sun_Latn',	\
            'Swedish': 'swe_Latn',	\
            'Swahili': 'swh_Latn',	\
            'Silesian': 'szl_Latn',	\
            'Tamil': 'tam_Taml',	\
            'Tatar': 'tat_Cyrl',	\
            'Telugu': 'tel_Telu',	\
            'Tajik': 'tgk_Cyrl',	\
            'Tagalog': 'tgl_Latn',	\
            'Thai': 'tha_Thai',	\
            'Tigrinya': 'tir_Ethi',	\
            'Tamasheq (Latin script)': 'taq_Latn',	\
            'Tamasheq (Tifinagh script)': 'taq_Tfng',	\
            'Tok Pisin': 'tpi_Latn',	\
            'Tswana': 'tsn_Latn',	\
            'Tsonga': 'tso_Latn',	\
            'Turkmen': 'tuk_Latn',	\
            'Tumbuka': 'tum_Latn',	\
            'Turkish': 'tur_Latn',	\
            'Twi': 'twi_Latn',	\
            'Central Atlas Tamazight': 'tzm_Tfng',	\
            'Uyghur': 'uig_Arab',	\
            'Ukrainian': 'ukr_Cyrl',	\
            'Umbundu': 'umb_Latn',	\
            'Urdu': 'urd_Arab',	\
            'Northern Uzbek': 'uzn_Latn',	\
            'Venetian': 'vec_Latn',	\
            'Vietnamese': 'vie_Latn',	\
            'Waray': 'war_Latn',	\
            'Wolof': 'wol_Latn',	\
            'Xhosa': 'xho_Latn',	\
            'Eastern Yiddish': 'ydd_Hebr',	\
            'Yoruba': 'yor_Latn',	\
            'Yue Chinese': 'yue_Hant',	\
            'Chinese (Simplified)': 'zho_Hans',	\
            'Chinese (Traditional)': 'zho_Hant',	\
            'Standard Malay': 'zsm_Latn',	\
            'Chinese': 'zho_Hans',	\
            'Zulu': 'zul_Latn'}
    lan_code = dict.get(lan, 'Not Found')
    if lan_code == "Not Found":
        raise ValueError(f"Language {lan} has not been supported!")
    return lan_code