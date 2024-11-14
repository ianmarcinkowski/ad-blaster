# Ad Blaster

## Using a local LLM to detect ads and send IR mute commands to a TV

<img width="1495" alt="Ian Marcinkowski - Hackathon - 2024-11-01 - Diagram" src="https://github.com/user-attachments/assets/7527469d-25de-4c9f-a422-3e32fda3addb">

### Intro blog post

When visiting my parents, the TV is usually on, and my parents have mostly stopped noticing advertising at this point.  It's just background noise to them.  Contrast that with my experience of using adblocking since browers have had plugins, and paying for things like Youtube Premium to avoid the ads.  

Most ads are powerless without sound, so I am usually placed in charge of the TV remote because I care the most about this problem and my parents can be saved the annoyance of having to constantly mute and unmute something they are barely even aware of.  

AI should be used to provide good outcomes for humans, and I think muting ads and taking their power away seems like a fine use!

**Keeping It Local**

Within the last few weeks (Oct 2024), Meta's `llama3.2-vision` model finally became available on the Ollama model library.  Ollama is a desktop app that allows AI models to be run locally and accessed via HTTP.  It could be run on my M2 Macbook or Windows desktop with a Radeon 7900XT GPU.

Keeping everything local also has the benefit of not sharing webcam footage from inside my home with outside parties who may not respect my privacy.  I assume that even if I'm paying OpenAI, I am still providing data to a company I don't have a lot of trust in.

Months ago, I experimented with OpenAI's API and was able to get reasonably accurate results from GPT-4 with image recognition, but found that the associated costs could reach tens of dollars per hour to make requests quickly enough to meaningfully mute the TV during an ad break.

**Collaborating on hardware with my partner**

My partner has an Arduino and was excited to have a concrete reason to experiment, so she took charge of the electronics side of the project. 

She wired up an IR receiver integrated-circuit to the Arduino, and used the `IRRemote` library to read the IR codes being sent by my TV's remote control.  Next, she began working on how to send the IR signals using the Arduino, and how to control when to send the signal via input from the Arduino's serial port so it could be integrated to the rest of the system.

On the last day of Hackathon, we encounted some uncertainty about outputting the IR signal using the `IRRemote` library, and being concerned about applying too much current to the IR LED and burning it out given that we didn't have the correct value resistors recommended on the Arduino forums.

Instead, my partner added an LED to be used as a visual indicator for the demo; red for muted, off for unmuted.


**Performance Stats**

At the start of this project, I quickly tested whether I could accomplish the goal only using my Macbook M2.  My quick initial tests showed that response generation took 10x longer on my Macbook than using the RX 7900XT in my desktop computer.

The *Tool Use Only* prompt asked for outputs in the format:
```
\```python
advertising_detected(True or False)
\```
```

The *Tool Use + Reason* prompt asked for outputs in the format:
```
\```python
advertising_detected(True or False)
Reason: A description of why advertising is likely.
\```
```


| Platform | Prompt            | Min (ms)    | Max (ms)    | Average (ms) | N   |
| -------- | ----------------- | ----------- | ----------- | ------------ | --- |
| 7900XT   | Tool Use + Reason | 4.8072046   | 13.519339   | 7.95031584   | 84  |
| Mac M2   | Tool Use + Reason | 60.93292871 | 79.67972733 | 63.01418841  | 30  |
| 7900XT   | Tool Use Only     | 2.5100168   | 3.4641747   | 2.553410857  | 68  |
| Mac M2   | Tool Use Only     | 57.96496817 | 68.27587954 | 59.1616763   | 30  |

* Reasonably accurate results in the Tool Use + Reasoning cases
* Tool use only case: Always False, not useful, prompt improvements for tool-use-only might help

**Now I understand why there are all those articles on prompt engineering**

My own prompts were not steering the model to an answer that I could easily use.

Initially, I was worried that longer prompts would require too much time for the model to process.  This proved to be a misconception which was cleared up by asking GPT-4o to generate the system and user prompts that I needed.  

These GPT-4o-generated prompts were much longer and included strongly worded directions to steer the model to producing helpful output.  They included bolded phrases to `**Keep the response short**` or `**Maintain Objectivity**`, and other calls to action, as well as examples of scenarios and expected output.

**Progress achieved during work Hackathon**

The core of the app captures images of the TV screen using `python-opencv` with a second webcam plugged into my Macbook.  This image is base64-encoded and sent to Ollama & `llama3.2-vision`.  After many iterations of different prompts, we landed on a reasonably good rate of detecting advertisements every 7-ish seconds.

Common mis-classifications happen during sports promos and on daytime TV shows that demo different products like the Kelly Clarkson show.

I learned that there is a tradeoff between 2-5 second, low-accuracy outputs, and 7-12 second medium-accuracy outputs. 

Further improvements to accuracy could include better prompt engineering to improve the accuracy of the Tool Use Only prompt, windowing that only mutes after several positive signals, and buying more GPUs to improve generation time.

**Post-hackathon progress**

A couple hours after the hackathon demos ended, we resolved the uncertainty with sending IR signals by Arduino and have a working proof of concept that mutes the TV when ads are detected! 

Our solution is still limited by the false-positive rate of our prompting techniques.

Overall, we are very happy with the result and work is ongoing to improve the accuracy of the ad detection.

