from pathlib import Path

class Config:

    # Model Configuration
    MAYA_CONFIGURATION = {
        "model_base": "CohereForAI/aya-23-8B",
        "model_path": "maya-multimodal/maya",
        "mode": "finetuned",
        "Maya_path": "./maya", # git clone https://github.com/nahidalam/maya
    }

    PANGEA_CONFIGURATION = {
        "model_base": None,
        "model_path": "neulab/Pangea-7B",
        "model_name": "Pangea-7B-qwen",
        "multimodal": True,
        "LLaVA-NeXT_path": "./LLaVA-NeXT" # git clone https://github.com/LLaVA-VL/LLaVA-NeXT
    }

    QWENVL_CONFIGURATION = {
        "model_path": "Qwen/Qwen2-VL-7B-Instruct",
        "min_pixels": 256*28*28,
        "max_pixels": 1280*28*28,
    }

    PALIGEMMA2_CONFIGURATION = {
        "model_path": "google/paligemma2-10b-ft-docci-448",
    }


    # Dataset Configuration
    WORLDCUISINE_CONFIGURATION = {
        "dataset_path": "worldcuisines/food-kb",
        "SEA_REGIONS": ["South Eastern Asia"], # only retrieve cuisine related to SEA
        "MAX_IMAGES_PER_ITEM": 8, # max image taken for each cuisines, default to all(8)
    }

    SEAVQA_CONFIGURATION = {
        "dataset_path": "./filter_sea_vqa_final", # this dataset contain metadata of SEA-VQA, and currently is not available for public
    }

    
    # Common prompt
    PROMPT_CONFIGURATION = {
        "en_location_agnostic_prompt": "Write a caption in English for an image that may include culturally significant objects or elements from Southeast Asia. The caption should specifically name Southeast Asian cultural items, such as cuisine, traditions, landmarks, or other related elements if they appear in the image. The caption should be concise, consisting of 3 to 5 sentences.",
        
        "th_location_agnostic_prompt": "เขียนแคปชั่นเป็นภาษาไทยสำหรับรูปที่อาจมีวัตถุทางวัฒนธรรมหรือองค์ประกอบจาก Southeast Asia แคปชั่นควรจะมีชื่อวัตถุวัฒนธรรมของ Southeast Asian ได้แก่ ชื่ออาหาร วัฒนธรรม สถานที่ หรืออะไรก็ตามที่ปรากฏบนรูปภาพ แคปชั่นควรมีความยาว 3 ถึง 5 ประโยค",
        
        "ms_location_agnostic_prompt": "Tulis kapsyen dalam bahasa Melayu untuk imej yang mungkin mengandungi objek atau unsur penting budaya dari Asia Tenggara. Kapsyen harus menyebut secara khusus item budaya Asia Tenggara, seperti makanan, tradisi, mercu tanda, atau elemen lain yang berkaitan jika ia muncul dalam imej. Kapsyen hendaklah ringkas, terdiri daripada 3 hingga 5 ayat.",
        
        "tl_location_agnostic_prompt": "Magbigay ng caption sa Tagalog para sa isang litrato na maaaring may elemento o aspetong makahulugan para sa Timog-Silangang Asya. Dapat pinapangalanan ng caption na ito ang mga cultural at tradisyunal na bagay sa Timog-Silangang Asya tulad ng pagkain, tradisyon, lugar, o anumang kaugnay na elemento kung ito ay kasama sa litrato. Ang caption ay dapat na maigsi, mga 3 hanggang 5 pangugusap lamang.",
        
        "id_location_agnostic_prompt": "Tulis deskripsi dalam bahasa Indonesia untuk gambar yang mungkin mengandung objek atau elemen budaya penting di Asia Tenggara. Deskripsi harus menyebutkan secara spesifik barang-barang dalam budaya Asia Tenggara, seperti makanan, tradisi, tempat bersejarah, atau elemen terkit lainnya jika mereka muncul dalam gambar. Deskripsi harus singkat, terdiri dari 3 sampai 5 kalimat.",
        
        "vi_location_agnostic_prompt": "Viết chú thích bằng tiếng Anh cho một hình ảnh mà nó có thể chứa các vật thể hoặc yếu tố văn hóa quan trọng của Đông Nam Á. Chú thích phải nêu tên cụ thể của các yếu tố văn hóa Đông Nam Á - chẳng hạn như ẩm thực, phong tục, địa danh hoặc các yếu tố liên quan khác - nếu chúng xuất hiện trong hình ảnh. Chú thích phải ngắn gọn, chỉ bao gồm từ 3 đến 5 câu.",
        
        "en_location_aware_prompt": "This is an image from {Location}. Write a caption in English for an image that may include culturally significant objects or elements from {Location}. The caption should specifically name Southeast Asian cultural items, such as cuisine, traditions, landmarks, or other related elements if they appear in the image. The caption should be concise, consisting of 3 to 5 sentences.",
        
        "th_location_aware_prompt": "นี้คือรูปจาก Thailand เขียนแคปชั่นในภาษาไทยสำหรับรูปที่อาจมีวัตถุทางวัฒนธรรมจาก Thailand แคปชั่นควรกล่าวถึงวัตถุวัฒนธรรมทาง Southeast Asian ได้แก่ ชื่ออาหาร วัฒนธรรม สถานที่ หรืออื่นๆที่อาจปรากฏบนรูป แคปชั่วควรมีความยาว 3 ถึง 5 ประโยค",
        
        "ms_location_aware_prompt": "Ini adalah imej dari Malaysia. Tulis kapsyen dalam bahasa Melayu untuk imej yang mungkin mengandungi objek atau unsur penting budaya dari Malaysia. Kapsyen harus menyebut secara khusus item budaya Asia Tenggara, seperti makanan, tradisi, mercu tanda, atau elemen lain yang berkaitan jika ia muncul dalam imej. Kapsyen hendaklah ringkas, terdiri daripada 3 hingga 5 ayat.",
        
        "tl_location_aware_prompt": "Ito ay isang litrato mula sa Philippines. Magsulat ng caption sa Tagalog para sa isang litrato na may elemento o bagay na makabuluhan sa kultura ng Philippines. Dapat pinapangalanan ng caption na ito ang mga cultural at tradisyunal na bagay sa Timog-Silangang Asya tulad ng pagkain, tradisyon, lugar, o anumang kaugnay na elemento kung ito ay kasama sa litrato. Ang caption ay dapat na maigsi, mga 3 hanggang 5 pangugusap lamang.",
        
        "id_location_aware_prompt": "Ini adalah gambar dari Indonesia. Tulis deskripsi dalam bahasa Indonesia untuk gambar yang mungkin mengandung objek atau elemen budaya penting dari Indonesia. Deskripsi harus menyebutkan secara spesifik barang-barang dalam budaya Asia Tenggara, seperti makanan, tradisi, tempat bersejarah, atau elemen terkit lainnya jika mereka muncul dalam gambar. Deskripsi harus singkat, terdiri dari 3 sampai 5 kalimat.",
        
        "vi_location_aware_prompt": "Đây là hình ảnh từ Vietnam. Hãy viết chú thích bằng tiếng Anh cho một hình ảnh mà nó có thể chứa các vật thể hoặc yếu tố văn hóa quan trọng của Vietnam. Chú thích phải nêu tên cụ thể của các yếu tố văn hóa Đông Nam Á - chẳng hạn như ẩm thực, phong tục, địa danh hoặc các yếu tố liên quan khác - nếu chúng xuất hiện trong hình ảnh. Chú thích phải ngắn gọn, chỉ bao gồm từ 3 đến 5 câu.",
    }

    RESULTS_DIR = Path("./results/") # replace it with absolute path
    
    # Default model parameters
    DEFAULT_GENERATION_CONFIG = {
        "max_new_tokens": 512, # Maximum sequence length
        "do_sample": False, # Enable sampling (default: False = greedy decoding)
        "num_beams": 1, # Beam search width
        "use_cache": True, # Enable transformer cache
        "temperature": None, # Sampling randomness (0-2)
        "top_p": None, # Nucleus sampling threshold (0-1)
        "top_k": None, # Top-k sampling threshold
    } # utilized greedy decoding for fairness comparison