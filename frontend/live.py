"""Controls and annotated pictures for the configured camera."""
import base64
import time

import httpx
import streamlit as st

from frontend.annotation import annotate

# The Streamlit server fetches the camera API, so browsers on another machine
# do not need access to the camera network or the backend's loopback address.
SOUND = st.components.v2.component(
    "live_alarm_sound", html='<button type="button">Enable alarm sound</button>',
    js="""export default function({parentElement, data}) {
      const button = parentElement.querySelector('button');
      const silence = () => {
        if (button.tone) { button.tone.stop(); button.tone = null; }
      };
      const update = () => {
        if (!data.active || !button.enabled) { silence(); return; }
        if (!button.tone && button.audio?.state === 'running') {
          const tone = button.audio.createOscillator();
          const gain = button.audio.createGain();
          tone.frequency.value = 880; gain.gain.value = 0.08;
          tone.connect(gain).connect(button.audio.destination);
          tone.start(); button.tone = tone;
        }
      };
      button.onclick = async () => {
        button.enabled = !button.enabled;
        button.audio ??= new AudioContext();
        await button.audio.resume();
        button.textContent = button.enabled ? 'Mute alarm sound' : 'Enable alarm sound';
        update();
      };
      update();
      clearTimeout(button.watchdog);
      button.watchdog = setTimeout(silence, 1500);
      return () => { button.onclick = null; };
    }""",
)


def live_view(settings):
    base = settings.live_backend_url.rstrip('/')

    def request(method, path, **kwargs):
        response = httpx.request(method, base + path, timeout=8, **kwargs)
        response.raise_for_status()
        return response

    @st.fragment(run_every=.25)
    def monitor():
        try:
            status = request('GET', '/status').json()
        except httpx.HTTPError:
            st.warning('Live backend unavailable.')
            return
        state = status['state']
        active = state not in ('stopped', 'error')
        if active != st.session_state.get('live_active', False):
            st.session_state.live_active = active
            st.rerun()
        if state == 'error':
            st.error(status['message'])
        if not active:
            return
        if st.button('Stop monitoring'):
            request('POST', '/stop')
        stale = time.time() - status.get('updated', 0) > 2
        if status['frame']:
            st.image(base64.b64decode(status['frame'].split(',', 1)[1]))
            if stale:
                st.warning('Waiting for a fresh camera picture.')
        else:
            st.info(status['message'])
        SOUND(data={'active': bool(status['active_alarm'] and not stale),
                    'updated': status.get('updated')}, key='live_sound')

    st.subheader('Live camera')
    try:
        monitor()
        if not st.session_state.get('live_active'):
            if st.button('Get camera preview'):
                st.session_state.live_preview = request('GET', '/preview').json()
            preview = st.session_state.get('live_preview')
            if preview:
                selection = annotate(preview)
                n = st.number_input('Minimum people inside the zone', min_value=1, value=1)
                seconds = st.number_input('More than this many seconds', min_value=.1, value=5.)
                points = (selection or {}).get('points', [])
                if st.button('Start monitoring', disabled=len(points) != 4):
                    request('POST', '/start', json=dict(polygon=points, n_people=n,
                            threshold_seconds=seconds, width=preview['width'], height=preview['height']))
                    st.rerun()
        if st.button('Load recent alarm events'):
            st.session_state.live_events = request('GET', '/events').json()
        if 'live_events' in st.session_state:
            events = st.session_state.live_events
            if events:
                event = st.selectbox('Saved alarms', events, format_func=lambda e: f"{e['utc']} · {e['reason']}")
                if st.button('View last alarm frame'):
                    st.image(request('GET', event['image_url']).content,
                             caption=f"Last active frame at {event['screenshot_seconds']:.2f} stream seconds")
            else:
                st.info('No saved alarms yet.')
    except httpx.HTTPStatusError as exc:
        st.error(str(exc.response.json().get('detail', 'Camera request failed.')))
    except httpx.HTTPError:
        st.error('Cannot reach the live backend.')
