import asyncio
import os
from pathlib import Path
from typing import List

from agents import Agent, Runner, RunContextWrapper
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from pydantic import BaseModel

import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
load_dotenv()    
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.5-flash'
# model_name = 'gemini-3.1-flash-lite'

def get_reflexion_solver_instructions(
    run_context: RunContextWrapper[str], agent: Agent[str]
) -> str:
    """Generate instructions for the reflexion solver agent."""
    instructions = (
        "You are a time-travel expert. Solve the problem step by step "
        "and be careful to avoid mistakes."
    )
    return instructions + "\nHINT:\n" + run_context.context


solver = Agent(
    name="TimeTravelerReflexion", 
    instructions=get_reflexion_solver_instructions,
    model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),
)
critic = Agent(
    name="TimeTravelCritic",
    instructions=(
        "You are an expert tutor. If the solution is wrong, explain the error "
        "and give a concise hint for improvement."
    ),
    model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
)

# --- Problem spec ------------------------------------------------------------
problem = (
    "I left the year 2000 in a time machine, went forward 30 years, "
    "then back 40 years. I claim I'm now in 1990. Am I correct?"
)
problem = """
In a sci-fi film, Alex is a time traveler who decides to go back in time
to witness a famous historical event that took place 100 years ago,
which lasted for 10 days. He arrives three days before the event starts.
However, after spending six days in the past, he jumps forward in time
by 50 years and stays there for 20 days. Then, he travels back to
witness the end of the end. 
How many days does Alex spend in the past before he sees the end of the event?
"""
TARGET_DAYS = "26"  # expected final answer
MAX_ATTEMPTS = 5  # fail-safe cap on retries

async def main():
    #solver loop
    feedback_hint = ""
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        result = await Runner.run(
            solver, 
            input=problem, 
            context=feedback_hint)
        answer = result.final_output.strip()
        print(f"\nAttempt {attempt_no}:\n{answer}")

        has_correct_days = TARGET_DAYS in answer
        says_claim_correct = "yes" in answer.lower() or "correct" in answer.lower()
        solved = has_correct_days and says_claim_correct

        if solved:
            print(" Solution accepted.")
            break

        feedback_prompt = (
            f"Solution given:\n{answer}\n\n"
            f"Expected final days: {TARGET_DAYS}\n"
            "Explain the error briefly and give a helpful hint."
        )
        feedback_resp = await Runner.run(critic, input=feedback_prompt)
        hint = feedback_resp.final_output.strip()

        print(f"Feedback:\n{hint}")
        feedback_hint = hint
    else:
        print("\n  Max attempts reached without a correct solution.")

if __name__ == "__main__":
    asyncio.run(main())


#############################################################
# Attempt 1:
# To determine how many days Alex spends in the past before witnessing the end of the event, we need to track his movements and the time he spends at each destination relative to his own timeline.

# 1.  **Initial Arrival and Stay:**
#     *   Alex travels back to a historical event that took place 100 years ago.
#     *   He arrives **3 days before** the event starts.
#     *   He stays in the past for **6 days**.
#     *   *Analysis:* During these 6 days, Alex spends 3 days waiting for the event to start and then witnesses the first 3 days of the 10-day event.
#     *   **Total days spent so far: 6 days.**

# 2.  **The First Jump:**
#     *   After his 6-day stay, he jumps forward in time by 50 years.
#     *   Because he was 100 years in the past, jumping forward 50 years places him **50 years in the past** relative to his original time.
#     *   He stays there for **20 days**.
#     *   *Analysis:* Since he is still in the "past" (50 years before his original timeline), these 20 days count toward his total time spent in the past.
#     *   **Total days spent so far: 6 + 20 = 26 days.**

# 3.  **The Second Jump (to witness the end):**
#     *   After spending 20 days in that time period, he travels back to witness the "end of the end" (the conclusion of the 10-day event).
#     *   The event lasts 10 days. The end of the event occurs 10 days after it started.
#     *   Alex travels directly to this moment.
#     *   *Analysis:* The question asks how many days he spends *before* he sees the end. Since the travel itself is instantaneous (unless specified otherwise) and he arrives specifically to witness the end, we do not add any more days to his personal count.

# **Conclusion:**
# Alex spent 6 days during his first stay and 20 days during his second stay, both of which were in the past relative to his origin.

# $6 \text{ days} + 20 \text{ days} = 26 \text{ days}$
# Feedback:
# The solution provided concludes that Alex spends **26 days** in the past, which is mathematically consistent with the individual stays mentioned ($6 + 20$). However, there is a subtle ambiguity in the interpretation of the timeline and the definition of "the past" relative to the event.

# ### The Error
# The error lies in the **Step 1 Analysis**. The solution assumes the **6-day stay** encompasses the **3 days** Alex spent waiting before the event started (3 days waiting + 3 days of the event). 

# However, in standard word problems of this type, the phrase "He arrives 3 days before... [and] he stays in the past for 6 days" often implies that the 6-day stay is the duration of the primary visit *once the event begins*, or that the waiting period and the stay are sequential. If the 6 days and 3 days are separate (3 days waiting + 6 days of the event), the total for the first leg would be **9 days**, leading to a final answer of **29 days**.

# Additionally, the solution fails to account for the **time spent witnessing the end**. If "witnessing the end" refers to the final day of the 10-day event, Alex spends an additional day in the past. While the question asks for the time *before* witnessing, the solution should clarify if he arrives at the *start* or the *end* of that final moment.

# ### Hint for Improvement
# To ensure the logic is airtight, clarify the relationship between the arrival and the stay:
# 1.  **Clarify Overlap:** Explicitly state whether the "6 days" is the **total duration** from the moment of arrival or the duration spent **at the event** after the 3-day wait.
# 2.  **Timeline Mapping:** Map the event days (e.g., Day -3 to Day 3) to show exactly which days of the 10-day event were seen. 
# 3.  **Personal vs. Global Time:** Ensure you are counting Alex's **proper time** (his personal clock) rather than the historical timeline, as his jump forward 50 years places his second stay chronologically *after* the event's end, even though it occurs *before* he witnesses it in his own timeline.

# **Final Check:** If the "6 days" is indeed the total stay, your addition ($6 + 20 = 26$) is correct, but the phrasing in the problem is often a trap to see if you add the 3-day arrival window to the subsequent stay.

# Attempt 2:
# To determine the total number of days Alex spends in the past before witnessing the end of the event, we must track his "proper time" (the time elapsed on his personal clock) across his various jumps.

# ### Step 1: The First Trip
# *   **Arrival:** Alex arrives 3 days before the event starts (let's call the start of the event Day 0). So, he arrives at Day -3.
# *   **Stay Duration:** The problem states he spends **6 days** in the past during this first leg. 
# *   **Timeline of First Stay:** Since he arrives at Day -3 and stays for 6 days, his stay covers:
#     *   Day -3, Day -2, Day -1 (3 days of waiting)
#     *   Day 1, Day 2, Day 3 (3 days of the event)
# *   **Total for Step 1:** **6 days**. (The 3-day waiting period is included in the "six days in the past" because he is in the past from the moment he arrives).

# ### Step 2: The Second Trip
# *   **The Jump:** Alex jumps forward 50 years. While this moves him through history, the jump itself is instantaneous for him.
# *   **Stay Duration:** He stays in this new time period (which is still 50 years in the "past" relative to his original starting point of 100 years ago) for **20 days**.
# *   **Total for Step 2:** **20 days**.

# ### Step 3: Determining the Time "Before" the End
# *   **The Goal:** Alex travels back to witness "the end of the end." The event lasted 10 days, so the "end" occurs on Day 10.
# *   **Chronological Logic:** The question asks for the total days Alex spent in the past **before** he sees the end. 
# *   His first stay (6 days) and his second stay (20 days) both occur in his personal timeline before he makes the final jump to the 10th day of the event.

# ### Step 4: Final Calculation
# To find the total time spent in the past before the final event, we add the duration of his first stay to the duration of his second stay:
# $$6 \text{ days (first stay)} + 20 \text{ days (second stay)} = 26 \text{ days}$$

# **Note on Interpretation:** Even though his second stay (50 years later) is chronologically *after* the event in historical time, it occurs *before* he witnesses the end of the event in his own personal time-traveling sequence. 

# Alex spends **26 days** in the past before he sees the end of the event.
# Feedback:
# The solution is logically sound in its final sum, but **Step 1 contains a chronological error** in how it labels the days. It identifies the arrival as "Day -3" and the event days as "Day 1, 2, 3," but skips **Day 0**. 

# If Alex arrives 3 days before an event and stays for 6 days, and we define the start as Day 0, his stay would cover:
# 1.  **Waiting period:** Day -3, -2, -1 (3 days)
# 2.  **Event period:** Day 0, 1, 2 (3 days)

# By listing the event days as "Day 1, 2, 3," the solution implies a 7-day span (including Day 0) or a gap in the timeline. While the total of 26 is correct based on the durations provided ($6 + 20$), the timeline used to explain the first 6 days is inconsistent.

# **Hint for improvement:**
# When mapping a timeline that crosses from "before" to "during" an event, remember to include **Day 0** as the starting point. If you arrive at the start of Day -3 and stay for 6 full days, you will leave at the start of Day 3, meaning you only witness three days of the event (0, 1, and 2). Ensure your day counts don't accidentally skip a day in the sequence!

# Attempt 3:
# To solve this problem accurately, we must map out the timeline of Alex’s travels day by day, ensuring we account for every day spent in the past and properly identify the starting point of the event.

# ### Step 1: The Initial Stay
# Alex arrives 3 days before a 10-day event. To maintain a clear timeline, we will designate the start of the event as **Day 0**.

# *   **Arrival Day:** Day -3
# *   **Duration of stay:** 6 days

# Alex’s stay follows this sequence:
# 1.  **Day 1:** Day -3 (3 days before)
# 2.  **Day 2:** Day -2 (2 days before)
# 3.  **Day 3:** Day -1 (1 day before)
# 4.  **Day 4:** Day 0 (**Event Starts**)
# 5.  **Day 5:** Day 1
# 6.  **Day 6:** Day 2

# At the end of these 6 days, Alex has witnessed the first 3 days of the event (Day 0, 1, and 2). 
# **Days spent so far: 6 days.**

# ### Step 2: The Forward Jump
# Alex then jumps forward in time by 50 years. Since the event took place 100 years before his original time, being 50 years in the future still places him in "the past" relative to his starting point.

# *   **Duration of stay in this period:** 20 days.

# **Days spent in this period: 20 days.**

# ### Step 3: Traveling to the End of the Event
# After these 20 days, Alex travels back to the original event to witness its conclusion. The question asks how many days he spent in the past **before** he sees the end of the event.

# We sum the durations of his stays:
# *   6 days (Initial arrival and start of the event)
# *   20 days (Stay 50 years in the future)

# **Total Calculation:**
# $6 \text{ days} + 20 \text{ days} = 26 \text{ days}$

# **Conclusion:**
# Alex spends a total of **26 days** in the past before he jumps back to witness the end of the historical event.
# Feedback:
# The solution provided is **correct**. It accurately maps out the timeline of Alex's stays and correctly sums the days spent in the past.

# ### Detailed Breakdown:
# 1.  **First Stay:** Alex arrives 3 days before the event and stays for a total of 6 days.
#     *   Days before the event: 3 days (e.g., Day -3, Day -2, Day -1).
#     *   Days of the event: 3 days (e.g., Day 0, Day 1, Day 2).
#     *   **Total for first stay: 6 days.**
# 2.  **Second Stay:** Alex jumps 50 years into the future. Since the original event was 100 years in the past, he is now 50 years before his original time. This period is still "the past."
#     *   **Total for second stay: 20 days.**
# 3.  **The Jump Back:** Alex travels back to witness the conclusion of the event. Since he is jumping to a specific point in time (the conclusion), he does not spend any additional days *before* seeing the end.
# 4.  **Final Calculation:** $6 \text{ days (Stay 1)} + 20 \text{ days (Stay 2)} = 26 \text{ days}$.

# **Hint for verification:** Always ensure that when a traveler "jumps" to a specific historical moment, you only count the biological days they experienced *at* each destination, not the years that passed in between. In this case, 6 experienced days + 20 experienced days equals 26 days.

# Attempt 4:
# To solve for the total number of days Alex spent in the past before witnessing the conclusion of the historical event, we need to track his biological time (the number of days he actually experienced) across his various jumps.

# ### 1. The First Stay (100 Years Ago)
# *   **The Event:** Lasts 10 days.
# *   **Arrival:** Alex arrives 3 days before the event starts.
# *   **Duration of Stay:** He stays for **6 days**.
# *   **Timeline of First Stay:**
#     *   Day 1-3: The three days leading up to the event.
#     *   Day 4-6: The first three days of the historical event.
# *   **Total days spent so far:** 6 days.

# ### 2. The Second Stay (50 Years Later)
# *   **The Jump:** Alex jumps 50 years into the future. Since the original event was 100 years in the past, he is now 50 years before his original starting "present" time. This period is still considered "the past."
# *   **Duration of Stay:** He stays for **20 days**.
# *   **Total days spent so far:** 6 days (Stay 1) + 20 days (Stay 2) = **26 days**.

# ### 3. The Final Jump
# *   **The Destination:** Alex travels back to the specific moment the event concludes (the "end of the end").
# *   **The Timing:** Because he travels directly to that specific moment in time, he does not spend any additional biological days traveling or waiting before he sees the conclusion.

# ### Final Calculation
# To find the total days Alex spent in the past before seeing the end of the event, we sum the duration of his stays:
# $$\text{Stay 1} + \text{Stay 2} = 6 \text{ days} + 20 \text{ days} = 26 \text{ days}$$

# Alex spends **26** days in the past before he sees the end of the event.
# Feedback:
# The solution provided correctly identifies the two main periods Alex spends in the past (6 days and 20 days) but contains a common logical error regarding the **final day of the event**.

# ### The Error
# The solution assumes that witnessing the "conclusion" of a 10-day event takes 0 biological days because Alex jumps "directly" to that moment. However, to **witness the conclusion** of an event that lasts 10 days (of which he has only seen the first 3), Alex must be biologically present for the final day (Day 10). Even if his travel is instantaneous, the act of experiencing that final day adds **one additional day** to his biological time spent in the past.

# Thus, the total should be the 6 days from the first stay, the 20 days from the second stay, and the 1 day spent witnessing the conclusion.

# ### Hint for Improvement
# Consider whether "witnessing" the final part of a historical event requires Alex to spend a full biological day there. If he has only seen Day 3 and jumps to the end of Day 10, he must still exist in the past for the duration of that final day to see it through.

# **Corrected Calculation:**
# $$6 \text{ days (Stay 1)} + 20 \text{ days (Stay 2)} + 1 \text{ day (Conclusion)} = 27 \text{ days}$$

# Attempt 5:
# To determine how many biological days Alex spends in the past, we break his journey down into three distinct segments of time spent in "the past" (relative to his original present).

# ### 1. The First Stay (Initial Arrival)
# Alex travels back to witness an event that occurred 100 years ago. 
# *   **Arrival Point:** He arrives **3 days before** the event starts.
# *   **Duration:** He stays for **6 days**.
# *   **Timeline Check:** Since he arrived 3 days before the event, after 6 days, he has witnessed the 3 days leading up to the event and the first 3 days of the 10-day event.
# *   **Biological Days Spent:** 6 days.

# ### 2. The Second Stay (The 50-Year Jump)
# Alex jumps forward 50 years (placing him 50 years before his original starting point).
# *   **Duration:** He stays in this time period for **20 days**.
# *   **Biological Days Spent:** 20 days.

# ### 3. Witnessing the Conclusion
# Alex travels back to witness the "end of the end" of the original 10-day event.
# *   **Context:** He previously left the event after Day 3. The event lasts 10 days. 
# *   **The "Witnessing" Requirement:** To witness the conclusion or the final day of a historical event, Alex must be biologically present for the duration of that final day. Even if his time travel is instantaneous, "witnessing" the end implies he exists in the past for that specific period.
# *   **Biological Days Spent:** 1 day.

# ### Total Calculation
# To find the total biological days Alex spends in the past, we sum the duration of each stay:
# $$6 \text{ days (Stay 1)} + 20 \text{ days (Stay 2)} + 1 \text{ day (Stay 3)} = 27 \text{ days}$$

# Alex spends a total of **27 days** in the past before he sees the end of the event.
# Feedback:
# The error in your calculation lies in including the duration of the final stay. The question asks for the total biological days Alex spends in the past **before** he sees the end of the event.

# **Hint for improvement:**
# Re-read the timing requirement carefully. Since you are calculating the time elapsed *up until the moment* he witnesses the conclusion, you should only sum the durations of his first two stays. The 1 day spent witnessing the "end of the end" occurs *during* the event you are using as the endpoint, not before it.

#   Max attempts reached without a correct solution.