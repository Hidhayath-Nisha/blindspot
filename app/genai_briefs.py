# ==============================================================================
# TRIAGE - LAYER 3: EMERGENCY BRIEF GENERATOR (LLM prompts)
# ==============================================================================
# This module generates the 3-audience briefs (Donor, UN, Journalist) 
# using an LLM API. It frames the intervention as a "Safety Warning" to fit 
# the SafetyKit hackathon track.
# ==============================================================================
import os
import json

def generate_safety_brief_prompts(crisis_data, optimized_lives_saved):
    """
    Constructs the exact LLM system prompts needed to generate the 3 briefs.
    In the Streamlit app, we will pass these strings to the OpenAI/Anthropic API.
    """
    country = crisis_data.get('iso3', 'Unknown Country')
    severity = crisis_data.get('Crisis_Severity_Score', 0)
    required = crisis_data.get('funding_required', 0)
    received = crisis_data.get('funding_received', 0)
    coverage = round((received / required) * 100) if required > 0 else 0
    
    # ---------------------------------------------------------
    # Audience 1: The Donor (Focus: ROI & Capital Efficiency)
    # ---------------------------------------------------------
    donor_prompt = f"""
    Act as a precise quantitative impact analyst. Write a 1-paragraph investment brief for a humanitarian donor. 
    Focus on ROI and capital efficiency.
    
    Data Context:
    - Crisis Location: {country}
    - Severity Score: {severity}/100 (Extremely High)
    - Current Funding Coverage: {coverage}%
    - Activating our optimized deployment algorithm saves a projected {optimized_lives_saved:,.0f} lives.
    
    Do not use emotional appeals. Frame this as a mathematically proven, high-ROI capital deployment opportunity 
    where early intervention prevents a massive physical safety collapse.
    """
    
    # ---------------------------------------------------------
    # Audience 2: The UN Coordinator (Focus: Benchmarking & Operations)
    # ---------------------------------------------------------
    un_prompt = f"""
    Act as a senior UN Operations Director. Write a 1-paragraph strategic brief for a field coordinator. 
    Focus on historical benchmarking and actionable operations.
    
    Data Context:
    - Crisis Location: {country}
    - Severity Score: {severity}/100 
    - Current Funding Coverage: {coverage}%
    
    State that this profile mirrors the pre-collapse signature of the 2011 Somalia famine. Advise them to immediately 
    allocate flexible pooled funds to the Health and WASH clusters to prevent the secondary physical safety wave (disease).
    """

    # ---------------------------------------------------------
    # Audience 3: The Journalist (Focus: The Invisible Crisis)
    # ---------------------------------------------------------
    journalist_prompt = f"""
    Act as an investigative data journalist. Write a 1-paragraph story pitch. 
    Focus on the "Invisible Crisis" narrative.
    
    Data Context:
    - Crisis Location: {country}
    - Severity Score: {severity}/100 (In the 90th percentile globally)
    - Current Funding Coverage: {coverage}%
    
    Highlight the mismatch: Describe how global media attention is focused strictly on one or two famous conflicts, 
    leaving the people in {country} quietly starving despite undeniable UN API data proving the physical danger.
    """
    
    # Instead of just returning the prompts, we will use Google Gemini to generate the actual briefs
    prompts = {
        "donor_prompt": donor_prompt,
        "un_prompt": un_prompt,
        "journalist_prompt": journalist_prompt
    }
    
    briefs = {
        "donor_brief": "Awaiting Generation...",
        "un_brief": "Awaiting Generation...",
        "journalist_brief": "Awaiting Generation..."
    }
    
    # Try to generate using Gemini API if the key is present
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            # Generate the 3 briefs
            response_donor = client.models.generate_content(model="gemini-2.5-flash", contents=donor_prompt)
            briefs["donor_brief"] = response_donor.text
            
            response_un = client.models.generate_content(model="gemini-2.5-flash", contents=un_prompt)
            briefs["un_brief"] = response_un.text
            
            response_journalist = client.models.generate_content(model="gemini-2.5-flash", contents=journalist_prompt)
            briefs["journalist_brief"] = response_journalist.text
            
        except ImportError:
            briefs["donor_brief"] = "Error: `google-genai` package not installed. Run: pip install google-genai"
            briefs["un_brief"] = "Error: `google-genai` package not installed."
            briefs["journalist_brief"] = "Error: `google-genai` package not installed."
        except Exception as e:
            error_msg = f"Gemini API Error: {str(e)}"
            briefs["donor_brief"] = error_msg
            briefs["un_brief"] = error_msg
            briefs["journalist_brief"] = error_msg
    else:
        # Fallback if no API key is set so the Hackathon demo doesn't crash
        briefs["donor_brief"] = f"[MOCK GENERATION - NO API KEY] {country} presents a high-ROI intervention opportunity. With a severity of {severity} and only {coverage}% funding, deploying capital here yields massive impact. Algorithm projects {optimized_lives_saved:,.0f} physical lives saved through rapid logistical scale-up."
        briefs["un_brief"] = f"[MOCK GENERATION - NO API KEY] The data signature in {country} mirrors pre-collapse Somalia. Immediate deployment of pooled contingency funds to the WASH and Health clusters is required to mitigate secondary physical safety casualties."
        briefs["journalist_brief"] = f"[MOCK GENERATION - NO API KEY] The Invisible Crisis: While the world watches elsewhere, data proves {country} is in the 90th percentile of physical danger with {coverage}% funding. The numbers don't lie, but the funding hasn't followed."

    return briefs

# Example usage for testing
if __name__ == "__main__":
    crisis_test = {
        'iso3': 'SDN', 
        'Crisis_Severity_Score': 92.5,
        'funding_required': 2700000000,
        'funding_received': 405000000
    }
    
    # Assuming the optimizer told us this allocation saves 14k lives
    lives = 14250 
    
    prompts = generate_safety_brief_prompts(crisis_test, lives)
    print("============= LLM PROMPTS GENERATED =============")
    print("DONOR:\n", prompts['donor_brief'])
    print("\nJOURNALIST:\n", prompts['journalist_brief'])
