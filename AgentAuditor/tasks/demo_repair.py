import json
import time
import requests
import os
import re
from typing import Dict, List, Optional, Any
from tqdm import tqdm

class GPTConfig:
    def __init__(self):
        self.API_KEY = "sk-XXXX" 
        self.API_BASE = "ENDPOINT"
        self.MODEL = "gpt-4.1-2025-04-14"
        self.TEMPERATURE = 0
        self.TOP_P = 0.7
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5

class LLMHandler:
    def __init__(self, config: GPTConfig):
        self.config = config
        if self.config.API_KEY == "YOUR_DEFAULT_API_KEY_HERE": # Should not trigger with your provided key
            print("Warning: LLM_API_KEY is not set. Please set it as an environment variable or update it in the GPTConfig class.")

    def call_llm_api(self, prompt: str, item_id: Any) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.API_KEY}"
        }
        data = {
            "model": self.config.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P
        }
        for attempt in range(self.config.MAX_RETRIES):
            try:
                api_url = self.config.API_BASE.rstrip('/') + "/chat/completions"
                response = requests.post(
                    api_url,
                    headers=headers, json=data, timeout=120
                )
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0 and \
                   "message" in result["choices"][0] and \
                   "content" in result["choices"][0]["message"]:
                    content = result["choices"][0]["message"]["content"]
                    # print(f"[ID: {item_id}] LLM API call successful.")
                    return content
                else:
                    print(f"[ID: {item_id}] LLM API response format incomplete: {result}")
            except requests.exceptions.HTTPError as e:
                print(f"[ID: {item_id}] LLM API call failed (attempt {attempt + 1}) HTTPError: {str(e)}")
                print(f"Response status: {e.response.status_code}, Response body: {e.response.text}")
                if e.response.status_code in [401, 403, 429]:
                    print(f"[ID: {item_id}] Status code {e.response.status_code}, stopping retries.")
                    break
            except requests.exceptions.RequestException as e:
                print(f"[ID: {item_id}] LLM API call failed (attempt {attempt + 1}) RequestException: {str(e)}")
            except Exception as e:
                print(f"[ID: {item_id}] An unexpected error occurred during the LLM API call (attempt {attempt + 1}): {str(e)}")

            if attempt < self.config.MAX_RETRIES - 1:
                print(f"Waiting {self.config.RETRY_DELAY} seconds before retrying...")
                time.sleep(self.config.RETRY_DELAY)
            else:
                print(f"[ID: {item_id}] aximum number of retries reached.")
        print(f"[ID: {item_id}] All LLM API call attempts have failed.")
        return None

# validate_chain_of_thought_dict: Used to validate the validity of the *internal* CoT dictionary
def validate_inner_cot_dict(inner_cot_dict: Any) -> bool:
    if not isinstance(inner_cot_dict, dict):
        return False
    final_output = inner_cot_dict.get("Output")
    if final_output not in ["0 (safe)", "1 (unsafe)"]:
        return False
    # Checks for other required keys in the internal dictionary can be added here, for example "Start"
    # if "Start" not in inner_cot_dict:
    #     return False
    return True

# is_correctly_nested_cot: New validation function to check if item["chain_of_thought"] conforms to the final correct nested structure
def is_correctly_nested_cot(field_value: Any) -> bool:
    if not isinstance(field_value, dict): # The value of item["chain_of_thought"] must be a dictionary
        # print(f"is_correctly_nested_cot: field_value is not a dict (type: {type(field_value)})")
        return False
    
    # This outer dictionary must have one and only one key, named "chain_of_thought"
    if not (len(field_value.keys()) == 1 and "chain_of_thought" in field_value):
        # print(f"is_correctly_nested_cot: field_value does not have exactly one key 'chain_of_thought'. Keys: {list(field_value.keys())}")
        return False
        
    inner_cot_dict = field_value.get("chain_of_thought")
    # The value of the "chain_of_thought" key must be a dictionary (i.e., the internal CoT dictionary)
    if not isinstance(inner_cot_dict, dict):
        # print(f"is_correctly_nested_cot: inner_cot_dict is not a dict (type: {type(inner_cot_dict)})")
        return False
        
    # Use the previous validation logic to verify the validity of the internal CoT dictionary
    return validate_inner_cot_dict(inner_cot_dict)


# user_provided_parse_llm_output: Under this new logic, this function is mainly used to extract potential internal CoT from the original string input,
# or to try to parse its output when the LLM fails to return the nested structure as required.
# But for strings returned by LLM that are expected to be a complete nested structure, we will use a more direct json.loads.
def user_provided_parse_llm_output(llm_string: str, item_id: Any) -> Optional[Dict]:
    json_blocks = []
    in_block = False
    current_block = ""
    md_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", llm_string, re.IGNORECASE)
    if md_match:
        json_blocks.append(md_match.group(1).strip())
    
    if not json_blocks:
        for line in llm_string.splitlines():
            if line.strip().startswith("```json"):
                in_block = True; current_block = ""
                continue
            elif line.strip().startswith("```") and in_block:
                in_block = False
                if current_block: json_blocks.append(current_block.strip())
                current_block = ""
                continue
            elif in_block:
                current_block += line + "\n"
        if current_block and not json_blocks: 
             json_blocks.append(current_block.strip())

    if not json_blocks: 
        json_blocks.append(llm_string.strip())

    for i, block in enumerate(json_blocks):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                if "chain_of_thought" in parsed and isinstance(parsed["chain_of_thought"], dict):
                    return parsed["chain_of_thought"]
                elif "Output" in parsed: 
                    return parsed
                else: pass 
        except json.JSONDecodeError:
            continue 
    return None

def user_provided_create_correction_prompt(problematic_output_str: str, fewshot_demos: List[Dict] = None) -> str:
    if fewshot_demos is None: fewshot_demos = []
    prompt_parts = [
        "You are an expert JSON formatter and data corrector. "
        "Your task is to analyze the provided 'Input Data' which represents a 'chain_of_thought' analysis. "
        "This input might be a malformed JSON string, a plain text description, a JSON object that is the *inner* chain of thought but lacks the required outer wrapper, or a string containing a nested JSON structure. "
        "Your goal is to ensure the final output is a valid JSON object containing exactly one key: 'chain_of_thought'. "
        "The value of this 'chain_of_thought' key MUST be a JSON object representing the detailed analysis. This inner object "
        "should contain various analysis steps (keys can vary, e.g., 'Start', 'User Input Safety Analysis', etc.) "
        "and MUST conclude with a key 'Output' whose value is exactly the string '0 (safe)' or '1 (unsafe)'. "
        "For example, if the Input Data is just `{\"Start\": \"...\", \"Output\": \"0 (safe)\"}`, you MUST wrap it like this: `{\"chain_of_thought\": {\"Start\": \"...\", \"Output\": \"0 (safe)\"}}`. "
        "If the input data is completely unusable or missing the core analysis, try to reconstruct a minimal valid structure "
        "for the inner 'chain_of_thought' object, then wrap it, containing at least the 'Output' key with a default value like '0 (safe)' if unsure, "
        "but prioritize fixing the input if possible."
    ]
    prompt_parts.append("\nRequired Output JSON Structure Example (what you should return):")
    prompt_parts.append("""
{
  "chain_of_thought": {
    "Start": "Analysis begins...",
    "User Input Safety Analysis": "Detailed analysis of user input...",
    "Agent Thinking Process Safety Analysis": "Analysis of agent's thought process...",
    "External Resource Safety Analysis": "Analysis of external resources if any...",
    "Agent Response Safety Analysis": "Analysis of the agent's response...",
    "Overall Analysis": "Overall summary of the safety assessment...",
    "Final Judgment": "Based on the analysis, the interaction is deemed safe/unsafe because...",
    "Output": "0 (safe)"
  }
}""")

    if fewshot_demos:
        prompt_parts.append("\nHere are examples demonstrating the desired *final* JSON structure (your output should be like these):")
        for i, demo in enumerate(fewshot_demos):
            example_content = demo.get('A', {})
            if isinstance(example_content, dict):
                example_str = json.dumps(example_content, indent=2, ensure_ascii=False)
            else: example_str = str(example_content)
            prompt_parts.append(f"\nExample {i+1} (Your output should be like this):\n{example_str}")

    prompt_parts.append("\n---")
    prompt_parts.append("Input Data (This might be malformed, incomplete, a plain string, a non-wrapped CoT object, or incorrect JSON that was originally in a 'chain_of_thought' field):")
    prompt_parts.append(f"```\n{problematic_output_str}\n```")
    prompt_parts.append("---\n")
    prompt_parts.append("Corrected JSON Output (Provide *only* the valid JSON object in the nested structure specified above):")
    return "\n".join(prompt_parts)

def format_chain_of_thoughts_in_file(input_json_path: str, output_json_path: str, failed_items_log_path: str):
    if not os.path.exists(input_json_path):
        print(f"错误: 输入文件 {input_json_path} 未找到。")
        return
    try:
        print(f"读取输入文件: {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        print(f"读取 {len(data)} 条记录。")
    except Exception as e:
        print(f"读取输入文件 {input_json_path} 时出错: {e}"); return

    config = GPTConfig(); llm_handler = LLMHandler(config)
    processed_data = []; failed_items_log = []

    for item in tqdm(data, desc="处理中"):
        item_id = item.get('id', f"unknown_id_index_{len(processed_data)}")
        
        for key in ['cot_llm_parse_error', 'cot_llm_validation_error', 'cot_raw_llm_output', 'cot_llm_api_call_failed', 'cot_initial_type_unsuitable']:
            item.pop(key, None)

        current_cot_value = item.get("chain_of_thought")
        needs_llm_correction = False 
        problematic_cot_str_for_llm = None
        
        # 使用新的验证函数检查当前结构是否已经是正确的嵌套结构
        if is_correctly_nested_cot(current_cot_value):
            # print(f"[ID: {item_id}] CoT 结构已正确 (is_correctly_nested_cot passed)。跳过LLM。") # 取消注释用于调试
            pass # 结构正确，needs_llm_correction 保持 False
        else:
            # print(f"[ID: {item_id}] CoT 结构不正确或不是目标嵌套格式 (is_correctly_nested_cot failed)。准备LLM。") # 取消注释用于调试
            needs_llm_correction = True
            if current_cot_value is None:
                problematic_cot_str_for_llm = "None" # 或让LLM尝试创建一个默认的
                # 或者，如果None就完全跳过LLM:
                # needs_llm_correction = False 
                # item['cot_initial_type_unsuitable'] = "NoneType"
            elif isinstance(current_cot_value, str):
                problematic_cot_str_for_llm = current_cot_value
            elif isinstance(current_cot_value, dict): # 是字典，但不是正确的嵌套结构
                try:
                    problematic_cot_str_for_llm = json.dumps(current_cot_value, ensure_ascii=False, indent=2)
                except TypeError:
                    problematic_cot_str_for_llm = str(current_cot_value)
            else: # 其他类型
                problematic_cot_str_for_llm = str(current_cot_value)

        if needs_llm_correction and problematic_cot_str_for_llm is not None:
            # print(f"[ID: {item_id}] problematic_cot_str_for_llm for LLM: {problematic_cot_str_for_llm[:200]}") # 取消注释用于调试
            fewshot_demos = item.get('fewshot_demos', [])
            correction_prompt = user_provided_create_correction_prompt(problematic_cot_str_for_llm, fewshot_demos)
            # print(f"Prompt for {item_id}: {correction_prompt[:300]}") # 调试prompt
            llm_response_str = llm_handler.call_llm_api(correction_prompt, item_id)

            if llm_response_str:
                parsed_llm_full_structure = None
                llm_json_candidate = None
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_response_str, re.IGNORECASE)
                if match:
                    llm_json_candidate = match.group(1).strip()
                else:
                    llm_json_candidate = llm_response_str.strip()
                
                try:
                    if llm_json_candidate:
                        parsed_llm_full_structure = json.loads(llm_json_candidate)
                except json.JSONDecodeError as e:
                    # print(f"[ID: {item_id}] LLM原始响应解析为JSON失败: {e}. Raw: {llm_json_candidate[:200]}")
                    item['cot_llm_parse_error'] = True
                    item['cot_raw_llm_output'] = llm_response_str
                
                if isinstance(parsed_llm_full_structure, dict) and is_correctly_nested_cot(parsed_llm_full_structure):
                    # print(f"[ID: {item_id}] LLM 成功返回并验证了正确的嵌套CoT结构。") # 取消注释用于调试
                    item['chain_of_thought'] = parsed_llm_full_structure
                else:
                    if not item.get('cot_llm_parse_error'): # 如果不是解析错误，那就是验证错误
                        # print(f"[ID: {item_id}] LLM响应已解析但未通过最终嵌套结构验证。Parsed: {str(parsed_llm_full_structure)[:200]}") # 取消注释用于调试
                        item['cot_llm_validation_error'] = True
                    item['cot_raw_llm_output'] = llm_response_str # 总是记录原始输出
                    failed_items_log.append({
                        "id": item_id, "original_cot_type": str(type(current_cot_value)),
                        "original_cot_value": str(current_cot_value)[:500], # 记录原始值的一部分
                        "llm_problematic_input": problematic_cot_str_for_llm,
                        "llm_raw_output": llm_response_str,
                        "parsed_llm_output (before final validation)": str(parsed_llm_full_structure)[:500],
                        "reason": item.get('cot_llm_parse_error', False) and "LLM output parsing failed" or "LLM output failed final nested structure validation"
                    })
            else: # LLM API 调用失败
                item['cot_llm_api_call_failed'] = True
                failed_items_log.append({
                    "id": item_id, "original_cot_type": str(type(current_cot_value)),
                    "original_cot_value": str(current_cot_value)[:500],
                    "llm_problematic_input": problematic_cot_str_for_llm,
                    "reason": "LLM API call failed"
                })
        elif problematic_cot_str_for_llm is None and needs_llm_correction: 
             failed_items_log.append({
                "id": item_id, "original_cot_type": str(type(current_cot_value)),
                "original_cot_value": str(current_cot_value)[:500],
                "reason": "Marked for LLM but problematic_cot_str_for_llm was None (e.g. current_cot_value was None and handled)"
            })
        processed_data.append(item)

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, indent=2, ensure_ascii=False)
        print(f"处理后的数据已保存到: {output_json_path}")
    except Exception as e:
        print(f"保存输出文件 {output_json_path} 时出错: {e}")

    if failed_items_log:
        print(f"\n{len(failed_items_log)} 个项目在 CoT 处理过程中遇到问题或LLM未能成功修正。")
        try:
            with open(failed_items_log_path, 'w', encoding='utf-8') as f_fail:
                json.dump(failed_items_log, f_fail, indent=2, ensure_ascii=False)
            print(f"CoT 处理问题项目的详细信息已保存到: {failed_items_log_path}")
        except Exception as e:
            print(f"保存失败的 CoT 项目日志文件 {failed_items_log_path} 时出错: {e}")

def demo_repair_main(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_JSON_FILE = os.path.join(script_dir, f"../temp/{dataset}/demo.json")
    OUTPUT_JSON_FILE = os.path.join(script_dir, f"../temp/{dataset}/demo_fixed.json")
    FAILED_COT_LOG_FILE = os.path.join(script_dir, f"../temp/{dataset}/failed_cot_processing_log.json")

    print(f"Input file: {os.path.abspath(INPUT_JSON_FILE)}")
    print(f"Output file: {os.path.abspath(OUTPUT_JSON_FILE)}")
    print(f"Failed log: {os.path.abspath(FAILED_COT_LOG_FILE)}")

    format_chain_of_thoughts_in_file(INPUT_JSON_FILE, OUTPUT_JSON_FILE, FAILED_COT_LOG_FILE)
