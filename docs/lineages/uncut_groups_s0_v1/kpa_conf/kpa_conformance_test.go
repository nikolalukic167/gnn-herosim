// KPA conformance (ii): replay a recorded per-function concurrency trace (physics-audit `kpa` rows) through Knative Serving's own
// decider, open loop. The DeciderSpec and the windows come from the release's own defaults (autoscaler ConfigMap defaults, a
// PodAutoscaler with containerConcurrency 1, MakeDecider and MakeMetric); observations go into the release's TimedFloat64Buckets
// exactly as collection.record does (proxied = 0). Env: KPA_TRACE (in), KPA_OUT (per-tick CSV), KPA_DEFAULTS (JSON).
package kpaconf

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric/noop"
	"go.uber.org/zap"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"knative.dev/serving/pkg/apis/autoscaling"
	autoscalingv1alpha1 "knative.dev/serving/pkg/apis/autoscaling/v1alpha1"
	"knative.dev/serving/pkg/autoscaler/aggregation"
	asconfig "knative.dev/serving/pkg/autoscaler/config"
	"knative.dev/serving/pkg/autoscaler/metrics"
	"knative.dev/serving/pkg/autoscaler/scaling"
	kparesources "knative.dev/serving/pkg/reconciler/autoscaling/kpa/resources"
	aresources "knative.dev/serving/pkg/reconciler/autoscaling/resources"
)

type buckets struct {
	stable, panic *aggregation.TimedFloat64Buckets
}

// As MetricCollector.StableAndPanicConcurrency with a scrape target set: ErrNoData when the stable buckets are empty.
func (b *buckets) StableAndPanicConcurrency(_ types.NamespacedName, now time.Time) (float64, float64, error) {
	if b.stable.IsEmpty(now) {
		return 0, 0, metrics.ErrNoData
	}
	return b.stable.WindowAverage(now), b.panic.WindowAverage(now), nil
}
func (b *buckets) StableAndPanicRPS(types.NamespacedName, time.Time) (float64, float64, error) {
	return 0, 0, nil
}

type counter struct{ ready int }

func (c *counter) ReadyCount() (int, error)    { return c.ready, nil }
func (c *counter) NotReadyCount() (int, error) { return 0, nil }

type row struct {
	K     string  `json:"k"`
	T     float64 `json:"t"`
	Fn    string  `json:"fn"`
	Obs   float64 `json:"obs"`
	Cur   int     `json:"cur"`
	Ready int     `json:"ready"`
}

var epoch = time.Unix(1_000_000_000, 0)

func TestKpaConformance(t *testing.T) {
	cfg, err := asconfig.NewConfigFromMap(map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	pa := &autoscalingv1alpha1.PodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Namespace: "herosim", Name: "fn",
			Annotations: map[string]string{autoscaling.ClassAnnotationKey: autoscaling.KPA}},
		Spec: autoscalingv1alpha1.PodAutoscalerSpec{ContainerConcurrency: 1, Reachability: autoscalingv1alpha1.ReachabilityReachable},
	}
	dec := kparesources.MakeDecider(pa, cfg)
	met := aresources.MakeMetric(pa, "svc", cfg)
	spec := dec.Spec
	defaults := map[string]any{
		"target": spec.TargetValue, "total": spec.TotalValue, "stable_window_s": spec.StableWindow.Seconds(),
		"panic_window_s": met.Spec.PanicWindow.Seconds(), "panic_window_pct": cfg.PanicWindowPercentage,
		"panic_threshold": spec.PanicThreshold, "max_scale_up_rate": spec.MaxScaleUpRate, "max_scale_down_rate": spec.MaxScaleDownRate,
		"scale_down_delay_s": spec.ScaleDownDelay.Seconds(), "activation_scale": spec.ActivationScale, "reachable": spec.Reachable,
		"target_burst_capacity": spec.TargetBurstCapacity, "initial_scale": spec.InitialScale, "metric": spec.ScalingMetric,
		"bucket_s": asconfig.BucketSize.Seconds(), "scale_to_zero_grace_s": cfg.ScaleToZeroGracePeriod.Seconds(),
		"aggregation": pa.Annotations[autoscaling.MetricAggregationAlgorithmKey],
	}
	if p := os.Getenv("KPA_DEFAULTS"); p != "" {
		b, _ := json.MarshalIndent(defaults, "", " ")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	in, out := os.Getenv("KPA_TRACE"), os.Getenv("KPA_OUT")
	if in == "" || out == "" {
		t.Skip("KPA_TRACE / KPA_OUT unset: defaults only")
	}
	fh, err := os.Open(in)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	of, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	defer of.Close()
	w := bufio.NewWriter(of)
	defer w.Flush()
	fmt.Fprintln(w, "fn,t,obs,cur,ready,valid,desired,stable_avg,panic_avg")
	logger := zap.NewNop().Sugar()
	type fnState struct {
		b *buckets
		c *counter
		s scaling.UniScaler
	}
	fns := map[string]*fnState{}
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	n := 0
	for sc.Scan() {
		var r row
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		if r.K != "kpa" {
			continue
		}
		st := fns[r.Fn]
		if st == nil {
			st = &fnState{b: &buckets{aggregation.NewTimedFloat64Buckets(met.Spec.StableWindow, asconfig.BucketSize),
				aggregation.NewTimedFloat64Buckets(met.Spec.PanicWindow, asconfig.BucketSize)}, c: &counter{ready: r.Ready}}
			sp := spec
			st.s = scaling.New(attribute.NewSet(), noop.NewMeterProvider(), "herosim", r.Fn, st.b, st.c, &sp)
			fns[r.Fn] = st
		}
		now := epoch.Add(time.Duration(r.T * float64(time.Second)))
		st.b.stable.Record(now, r.Obs)
		st.b.panic.Record(now, r.Obs)
		st.c.ready = r.Ready
		res := st.s.Scale(logger, now)
		sa, pv := st.b.stable.WindowAverage(now), st.b.panic.WindowAverage(now)
		fmt.Fprintf(w, "%s,%.17g,%.17g,%d,%d,%t,%d,%.17g,%.17g\n", r.Fn, r.T, r.Obs, r.Cur, r.Ready, res.ScaleValid, res.DesiredPodCount, sa, pv)
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("ticks %d functions %d", n, len(fns))
}
